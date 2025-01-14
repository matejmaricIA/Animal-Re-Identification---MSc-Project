from rembg import remove

input_path = "image1.png"
output_path = "./out1.png"

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)
