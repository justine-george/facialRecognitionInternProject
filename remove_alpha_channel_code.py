from PIL import Image

face = Image.open('dataset/suriyasample.png')
face.load() #required for png split

background = Image.new("RGB", face.size, (255, 255, 255))
background.paste(face, mask=face.split()[3]) # 3 is the alpha channel

background.save('dataset/suriyasamplenew.jpg', 'JPEG', quality=80)
