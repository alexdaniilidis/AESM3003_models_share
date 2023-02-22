#########################################################################
# Parser for reading in .GRDECL files generated in RRM                  #                                                                       #
# Largely based on PyGRDECL Code by Bin Wang from U Louisiana           #
#########################################################################

import numpy as np

# keywords that are exported in RRM
SupportKeyWords=[
    'SPECGRID',
    'GRIDUNIT',
 #   'DX','DY','DZ',
    'COORD','ZCORN',
    'PORO', 
    'PERMX' , 'PERMY', 'PERMZ',
    'PRESSURE',
	'OPERNUM', 'ACTNUM'
]

# Corrsponding data types
KeyWordsDatatypes=[
    int,
    int,
#    float,float,float,
    float,float,
    float,
    float,float,float,
    float,
	int, int
]

class RRMParser:
    def __init__(self,filename=''):

        # setting up arrays
        self.fname=filename
        self.NX=0
        self.NY=0
        self.NZ=0
        self.N=0
        self.GRID_type='Unspecified'
        self.GRID_unit = 'Unspecified'

        # cartesian gridblock data KeyWords
        self.DX=[]
        self.DY=[]
        self.DZ=[]

        # Corner point gridblock data
        self.COORD=[]
        self.ZCORN=[]

        # Keywords
        self.SpatialDatas={}

        # Read GRDECL file when initializing the class
        if(len(filename)>0):
            self.read_GRDECL()
    
    ######[read_GRDECL]######
    def read_GRDECL(self):

        debug=0

        print('     Parsing GRDECL file generated in RRM with name \"%s\" ....'%(self.fname))

        # Read whole file into list
        f=open(self.fname)
        contents=f.read()
        contents=RemoveCommentLines(contents,commenter='--')
        contents_in_block=contents.strip().split('/') # Separate input file by slash /
        contents_in_block = [x for x in contents_in_block if x]# Remove empty block at the end
        NumKeywords=len(contents_in_block)
        print('     Total number of keywords read in file = %s' % (NumKeywords))

        GoodFlag=0
        ReadZCORN=0
        ReadACTNUM=0

        for i,block in enumerate(contents_in_block):#Keyword, Block-wise
            blockData_raw=block.strip().split()
            Keyword=''
            DataArray=[]
            if(len(blockData_raw)>1):
                if(blockData_raw[0]=='ECHO'): #This keyword may next to real keyword
                    Keyword,DataArray=blockData_raw[1],blockData_raw[2:]
                else:
                    Keyword,DataArray=blockData_raw[0],blockData_raw[1:]

            # Read grid unit
            if(Keyword=='GRIDUNIT'):
                DataArray = np.array(DataArray[:1])
                self.GRID_unit=DataArray[0]
                print('     Grid unit %s ' % (self.GRID_unit))
                print('     Successfully read keyword [%s] ' % (Keyword))
                continue

            # Read grid dimension
            if(Keyword=='SPECGRID'):
                DataArray=np.array(DataArray[:3],dtype=int)
                self.GRID_type='CARTESIAN' # RRM always generates Cartesian grids
                self.NX,self.NY,self.NZ=DataArray[0],DataArray[1],DataArray[2]
                self.N=self.NX*self.NY*self.NZ
                print("     Grid dimension (NX,NY,NZ): (%s x %s x %s)"%(self.NX,self.NY,self.NZ))
                print("     Number of grid cells = %s" %(self.N))
                print('     Successfully read keyword [%s] ' %(Keyword))
                GoodFlag=1
                continue
            
            #Read Grid spatial information, x,y,z ordering
            if(Keyword=='COORD'):# Pillar coords			
                assert len(DataArray)==6*(self.NX+1)*(self.NY+1),'[Error] Incompatible COORD data size!'
                self.COORD=np.array(DataArray,dtype=float)  
                print('     Successfully read keyword [%s] ' %(Keyword))
            elif(Keyword=='ZCORN'):# Depth coords
                if(len(DataArray)!=8*self.N): # in case GRDECL file uses, e.g. 128*0 128*10 128*20 etc. for ZCORN instead of full array                    
                    def increaseSize(lst, N):  #short function to append a list
                        return [el for el in lst for _ in range(N)]                    
                    #print("     Warning: Detected array size for ZCORN %s " % len(DataArray))
                    #print("     Expected array size for ZCORN is %s " % (8*self.N))
                    #print("     Checking if ECLIPSE shorthand for ZCORN is used and trying to convert")
                    text = DataArray[0].split('*') # convert first ZCORN entry (i.e. 128) to int and check that array lengths are consistent 
                    size = int(text[0])
                    assert len(DataArray)*size==8*self.N, '[Error] Incompatible ZCORN data size!'
                    length = len(DataArray)
                    list = [0.0]*(length) # create temporary list with ZCORN entries
                    for i in range(0,length):
                        text = DataArray[i].split('*')
                        val = float(text[1])
                        list[i] = val					
                    list = increaseSize(list, size)	# increase length of list by addind 127 entries of the same time for each entry
                    self.ZCORN = np.array(list)
                    ReadZCORN = 1 # set flag that ZCORN was read successfully
                elif(ReadZCORN==1): # in case RRM output is changed in the future and full array of ZCORN is specified
                    assert len(DataArray)==8*self.N, '[Error] Incompatible ZCORN data size!'
                    self.ZCORN=np.array(DataArray,dtype=float)
                print('     Successfully read keyword [%s] '%(Keyword))
            #Read Grid Properties information
            else:
                self.LoadVar(Keyword,DataArray,DataSize=self.N)
                if (Keyword == 'ACTNUM'):
                    ReadACTNUM = 1

        if (ReadACTNUM==0): # In cases where RRM models have flat surface, ACTNUM might not be written to file, setting ACTNUM for all grid blocks to 1
            print('     Keyword ACTNUM not found, setting all blocks to 1')
            KeywordID = SupportKeyWords.index('ACTNUM')
            self.SpatialDatas['ACTNUM'] =  np.array(np.ones(self.N),dtype=KeyWordsDatatypes[KeywordID])

        f.close()
        assert GoodFlag==1,'Can not find grid dimension info [SPECGRID]!'
        print('.....Done!')
    
    def LoadVar(self,Keyword,DataArray,DataSize):
        if(Keyword in SupportKeyWords):#KeyWords Check
            assert len(DataArray)==DataSize,'\n     [Error] Incompatible data size!'
            KeywordID=SupportKeyWords.index(Keyword)			
            self.SpatialDatas[Keyword]=np.array(DataArray,dtype=KeyWordsDatatypes[KeywordID])
            print('     Successfully read keyword [%s] '%(Keyword))

        else:
            print('     Warning: Unsupported keyword [%s]' % (Keyword))

    def ReturnData(self,name=''):
        if (name in SupportKeyWords):
            if (name=='NX'):
                return self.NX
            elif (name=='NY'):
                return self.NY
            elif (name=='NZ'):
                return self.NZ
            elif (name == 'ZCORN'):
                return self.ZCORN
            elif (name == 'COORD'):
                return self.COORD
            elif (name == 'GRIDUNIT'):
                return self.GRID_unit
            elif (name == 'SPECGRID'):
                return self.GRID_type
            else:
                return self.SpatialDatas[name]
        else:
            print('     Warning: Unsupported keyword [%s]' % (name))
            return name

    def ReturnNXYNYZ(self):
            return [self.NX,self.NY,self.NZ]

def RemoveCommentLines(data,commenter='--'):
    #Remove comment and empty lines
    data_lines=data.strip().split('\n')
    newdata=[]
    for line in data_lines:
        if line.startswith('--') or not line.strip():
            # skip comments and blank lines
            continue   
        newdata.append(line)
    return '\n'.join(newdata)

