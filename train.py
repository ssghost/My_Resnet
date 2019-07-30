import getopt, sys
import resnet

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:r:c:', ['images=','labels=','resize=','crop='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    ipath,lpath = '',''
    resize,crop = [0,0],[0,0]

    for o, a in opts:
        if o in ('-i', '--images') and type(a)==str:
            ipath = a
        elif o in ('-l', '--labels') and type(a)==str:
            lpath = a 
        elif o in ('-r', '--resize') and type(a)==int:
            resize = [a,a]
        elif o in ('-c', '-crop') and type(a)==int:
            crop = [a,a]
        else:
            assert False, 'unhandled option'
    
    res = resnet.resnet()
        
    res.input(ipath=ipath,lpath=lpath,resize=resize,crop=crop)
    res.compile_model()
    res.train()
    
if __name__ == "__main__":
    main()
