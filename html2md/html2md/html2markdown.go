package main
//package htmltomarkdown

/*
#include <stdlib.h>
*/
import "C"

import (
    "github.com/JohannesKaufmann/html-to-markdown/v2/converter"
    "github.com/JohannesKaufmann/html-to-markdown/v2/plugin/base"
    "github.com/JohannesKaufmann/html-to-markdown/v2/plugin/commonmark"
    "unsafe"
)

func NewConverter() *converter.Converter {
    return converter.NewConverter(
        converter.WithPlugins(
            base.NewBasePlugin(),
            commonmark.NewCommonmarkPlugin(),
        ),
    )
}



//export ConvertHTMLToMarkdown
func ConvertHTMLToMarkdown(html *C.char) *C.char {
    conv := NewConverter()
    markdown, _ := conv.ConvertString(C.GoString(html))
    return C.CString(markdown)
}


//export FreeString
func FreeString(s *C.char) {
	C.free(unsafe.Pointer(s))
}

func main() {}

