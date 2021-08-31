import re,string
import argparse
from xml.dom.minidom import parse
import xml.dom.minidom


parser = argparse.ArgumentParser()

parser.add_argument('--txt', type=str, default='/mango/homes/ZHAO_Ming/xfmt/zh-en/test.out.vanilla.txt')
parser.add_argument('--xml', type=str, default='/mango/homes/ZHAO_Ming/xfmt/test/testsrc.xml')
args = parser.parse_args()


pre_out = []
with open(args.txt, encoding='utf-8') as f:
    for snt in f.readlines():
        pre_out.append(snt.rstrip())
    
DOMTree = xml.dom.minidom.parse(args.xml)
collection = DOMTree.documentElement

collection.setAttribute("id", "testhyp")
collection.getElementsByTagName('DOC')[0].setAttribute("id", "oral")
collection.getElementsByTagName('src')[0].tagName="hyp"
collection.getElementsByTagName('hyp')[0].setAttribute("lang", "en")


segs = collection.getElementsByTagName("seg")
for i, seg in enumerate(segs):
    seg.childNodes[0].data = pre_out[i]
    
with open(args.txt + '.xml', 'w') as f:
    DOMTree.writexml(f, addindent='  ', encoding='utf-8')