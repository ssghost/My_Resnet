import getopt, sys
from resnet import *

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:r:c:m:', ['images=','outdir=','resize=','crop=','loadmodel='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    ipath,opath = '',''
    resize,crop = [0,0],[0,0]
    loadpath = None

    for o, a in opts:
        if o in ('-i', '--images') and type(a)==str:
            ipath = a
        elif o in ('-o', '--outdir') and type(a)==str:
            opath = a 
        elif o in ('-r', '--resize') and type(a)==int:
            resize = [a,a]
        elif o in ('-c', '--crop') and type(a)==int:
            crop = [a,a]
        elif o in ('-m', '--loadmodel') and type(a)==str:
            loadpath = a
        else:
            assert False, 'unhandled option'    
    
    if loadpath != None:
        resnet().load_model(loadpath)
        
    resnet().input(ipath=ipath,lpath=None,resize=resize,crop=crop)
    resnet().test(opath)
    
if __name__ == "__main__":
    main()
