from fasthtml.common import * 

from passlib.context import CryptContext

from functools import wraps

custom_styles = Style("""
.mw-960 { max-width: 960px; }
.mw-480 { max-width: 480px; }
.mx-auto { margin-left: auto; margin-right: auto; }

""")

app, rt = fast_app(live=True, debug=True, hdrs=(custom_styles,))

# all future code goes in here
db = database('data/users.db')

users = db.t.users

if users not in db.t:
    users.create(dict(email=str, password=str), pk='email')
    
User = users.dataclass()

def MyForm(btn_text, target):
    return Form(
        Input(id="email", type="email", placeholder="Email", required=True),
        Input(id="password", type="password", placeholder="Password", required=True),
        Button(btn_text, type="submit"),
        Span(id="error", style="color:red"),
        hx_post=target,
        hx_target="#error",
    )

@rt('/register')
def get():
    return Container(
        Article(
            H1("Register"),
            MyForm("Register", "/register"),
            Hr(),
            P("Already have an account? ", A("Login", href="/login")),
            cls="mw-480 mx-auto"
        )
    )


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

@rt('/register')
def post(email:str, password:str):
    
    try:
        users[email]
        return "User already exists"
    except NotFoundError:

        new_user = User(email=email, password=get_password_hash(password))
    
        users.insert(new_user)
        
        return HttpHeader('HX-Redirect', '/login')

@rt('/login')
def get():
    return Container(
        Article(
            H1("Login"),
            MyForm("Login", target="/login"),
            Hr(),
            P("Want to create an Account? ", A("Register", href="/register")),
            cls="mw-480 mx-auto"
        )
    )


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

@rt('/login')
def post(session, email:str, password:str):
    try:
        user = users[email]
    except NotFoundError:
        return "Email or password are incorrect"
    
    if not verify_password(password, user.password):
        return "Email or password are incorrect"

    session['auth'] = user.email
    
    return HttpHeader('HX-Redirect', '/dashboard')


def basic_auth(f):
    @wraps(f)
    def wrapper(session, *args, **kwargs):
        if "auth" not in session:
            return Response('Not Authorized', status_code=401)
        return f(session, *args, **kwargs)
    return wrapper


@rt('/dashboard')
@basic_auth
def get(session):    
    return Container(
        H1("Dashboard"),
        Button("Logout", hx_post="/logout")
    )

@rt('/logout')
def post(session):
    del session['auth']
    return HttpHeader('HX-Redirect', '/login')


serve()
