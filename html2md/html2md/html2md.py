from cffi import FFI

import os
lib_path = os.path.join(os.path.dirname(__file__), 'htmltomarkdown.so')

ffi = FFI()
ffi.cdef("""
    char* ConvertHTMLToMarkdown(char* html);
    void FreeString(char* s);
""")

#lib = ffi.dlopen("./htmltomarkdown.so")
lib = ffi.dlopen(lib_path)

def convert_html_to_markdown(html):
    if isinstance(html, str):
        html = html.encode('utf-8')
    result = lib.ConvertHTMLToMarkdown(html)
    markdown = ffi.string(result).decode('utf-8')
    lib.FreeString(result)
    return markdown

