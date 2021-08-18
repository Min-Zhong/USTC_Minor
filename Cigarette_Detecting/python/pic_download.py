import os
import sys
import requests
from threading import Thread
import re
import time
import hashlib

class BaiDu:
    def __init__(self, name, page, path):
        self.start_time = time.time()
        self.name = name
        self.page = page
        self.path = path
        self.url = 'https://image.baidu.com/search/acjson'
        self.header = {}
        self.num = 0

    def queryset(self):
        pn = 0
        for i in range(int(self.page)):
            pn += 60 * i
            name = {'word': self.name, 'pn': pn, 'tn':'resultjson_com', 'ipn':'rj', 'rn':60}
            url = self.url
            self.getrequest(url, name)

    def getrequest(self, url, data):
        print('[INFO]: Begin sending requests...' + url)
        ret = requests.get(url, headers=self.header, params=data)

        if str(ret.status_code) == '200':
            print('[INFO]: request 200 ok :' + ret.url)
        else:
            print('[INFO]: request {}, {}'.format(ret.status_code, ret.url))

        response = ret.content.decode()
        img_links = re.findall(r'thumbURL.*?\.jpg', response)
        links = []
        
        for link in img_links:

            links.append(link[11:])

        self.thread(links)

    def saveimage(self, link, path):
        print('[INFO]: Saving image...')
        m = hashlib.md5()
        m.update(link.encode())
        name = m.hexdigest()
        ret = requests.get(link, headers = self.header)
        image_content = ret.content
        filename = path + name + '.jpg'

        with open(filename, 'wb') as f:
            f.write(image_content)

        print('[INFO]:Saved sucessfully! {}.jpg'.format(name))

    def thread(self, links):
        self.num +=1
        for i, link in enumerate(links):
            print('*'*50)
            print(link)
            print('*' * 50)
            if link:
                # time.sleep(0.5)
                t = Thread(target=self.saveimage, args=(link,self.path))
                t.start()
                # t.join()
            self.num += 1
        print('All together {} requests.'.format(self.num))

    def __del__(self):

        end_time = time.time()
        print('Time used: {}(seconds)'.format(end_time - self.start_time))

def main():
    args = sys.argv[1:]
    if (len(args) < 3): 
        print ('Argument(s) missed!')
    name = args[0]
    num = args[1]
    path = args[2]
    baidu = BaiDu(name, num, path)
    baidu.queryset()


if __name__ == '__main__':
    main()
