with open('posts.txt', 'r') as f:
    text = f.read()
    cleaned_text = ''.join([i if ord(i) < 128 else '' for i in text])

with open('cleaned_posts.txt', 'w') as f2:
    f2.write(cleaned_text)