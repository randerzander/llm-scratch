Based on:
https://github.com/JohannesKaufmann/html-to-markdown

git clone https://github.com/JohannesKaufmann/html-to-markdown.git
cd html-to-markdown
cp ../html2markdown.go .

go build -buildmode=c-shared -o htmltomarkdown.so html2markdown.go
cp htmltomarkdown.so ../htmltomarkdown.so
