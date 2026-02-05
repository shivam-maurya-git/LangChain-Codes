from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
# Online Markdown Editor - The Best Free Markdown Tool 🚀

Experience the **fastest**, *most intuitive*, and ~~hassle-free~~ Markdown editor online!  
Create and preview Markdown instantly with **GitHub Flavored Markdown (GFM)** support.  

## ✨ Features of Online Markdown Editor

- **Live Preview**: Instantly see how your Markdown renders  
- **Auto-save**: Never lose your work with local storage backup  
- **File Management**: Create, edit, rename, and delete files easily  
- **Text Formatting**: Supports **bold**, *italic*, ~~strikethrough~~, <sup>superscript</sup>, and <sub>subscript</sub>  
- **Lists**: Easily create **bullet lists** and **numbered lists**  
- **Code Blocks**: Format your code with syntax highlighting  
- **Tables**: Create structured data with Markdown tables  
- **Mermaid Diagrams**: Visualize concepts with flowcharts and graphs  
- **Image & Link Insertion**: Easily add images and links  
- **Print & Download**: Save as a Markdown file or print directly  

---

## 📌 Markdown Syntax Guide  

### Headings  

# H1 - Largest Heading  
## H2 - Second Largest  
### H3 - Subheading  
#### H4 - Smaller Heading  
##### H5 - Tiny Heading  
###### H6 - Smallest Heading  

### ✍️ Text Formatting  

- **Bold** → `**Bold**` → **Bold**  
- *Italic* → `*Italic*` → *Italic*  
- ~~Strikethrough~~ → `~~Strikethrough~~` → ~~Strikethrough~~  
- <sup>Superscript</sup> → `<sup>Superscript</sup>`  
- <sub>Subscript</sub> → `<sub>Subscript</sub>`  
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 300,
    chunk_overlap = 0
    )
docs = splitter.split_text(text)
print(docs)
print(len(docs))