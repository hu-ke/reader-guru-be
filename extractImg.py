import PyPDF2
from pdf2image import convert_from_path

def extract_cover(pdf_path, output_image):
    # 读取PDF文件
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # 获取第一页
        page = reader.pages[0]
        # 将第一页保存为临时PDF文件
        temp_pdf = "temp.pdf"
        writer = PyPDF2.PdfWriter()
        writer.add_page(page)
        with open(temp_pdf, 'wb') as temp_file:
            writer.write(temp_file)
    
    # 将临时PDF文件转换为图像
    images = convert_from_path(temp_pdf)
    print(images[0])
    # 保存第一页图像
    images[0].save(output_image, 'JPEG')
    
    # 删除临时PDF文件
    import os
    os.remove(temp_pdf)

# 提取PDF的第一页为封面图片
# extract_cover("./charlie-and-the-chocolate-factory-by-roald-dahl.pdf", "book_covers/cover.jpg")