import getopt, sys
from resnet import Resnet

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:r:c:m:', ['inpath=','outpath=','resize=','crop=','modelpath='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    ipath,opath = '',''
    resize,crop = [0,0],[0,0]
    loadpath = None

    for o, a in opts:
        if o in ('-i', '--inpath') and type(a)==str:
            ipath = a
        elif o in ('-o', '--outpath') and type(a)==str:
            opath = a 
        elif o in ('-r', '--resize') and type(a)==int:
            resize = [a,a]
        elif o in ('-c', '--crop') and type(a)==int:
            crop = [a,a]
        elif o in ('-m', '--modelpath') and type(a)==str:
            modelpath = a
        else:
            assert False, 'unhandled option'    
    
    res = Resnet()
    if loadpath != None:
        res.load_model(modelpath)
        
    res.input(ipath=ipath,lpath=None,resize=resize,crop=crop)
    res.test(opath)
    
if __name__ == "__main__":
    main()
