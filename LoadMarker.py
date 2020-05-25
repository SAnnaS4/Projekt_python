import numpy as np
from struct import *

# -*- coding: iso-8859-1 -*-

class LoadMarker:
    def __init__(self, file_address):
        self.file_address = file_address
    def load(self):
        data = open(self.file_address, "rb").read()
        Left = np.zeros( 10)
        Top = np.zeros(10)
        Radius = np.zeros(10)
        index=np.zeros(10)
        #datanew=data[4:]
        if self.file_address.endswith(".mk1"):
            data_noinfo = 4
            data_length = 29

            for i in range(10):
                datastruct = '>dddIb'
                if i == 0:
                    s = unpack(datastruct, data[data_noinfo:(29 + data_noinfo)])
                else:

                    if i==1 :

                        s = unpack(datastruct, data[(i * (30 + 7)) + data_noinfo:(29 + (i * (30 + 7)) + data_noinfo)])
                    else:
                        s = unpack(datastruct, data[(((i-1) * (30 + 9))) + 30+7+data_noinfo:(29 + ((i-1) * (30 + 9)) +30+7+ data_noinfo)])

                index[i] = s[0]
                Left[i] = s[1]
                Top[i] = s[2]
                Radius[i] = s[4]


            data_totallength = len(data)

            data_list = data[(29 + (10 * (30 + 9)) + data_noinfo) + 98:]

            string_Marker = data.decode('cp855', errors='ignore')

            MarkerName = string_Marker.split()


            MarkerNameNew1 = []
            BeginMarker = 0
            start_byte=300
            for i in range(start_byte,len(data)):

                string_Marker = chr(data[i])
                if BeginMarker==1 and string_Marker=='?':
                    MarkerNameNew1.append('  ')


                if str.isalpha(string_Marker) and string_Marker!='ÿ':
                    # print(data[i-1])
                    if (((data[i - 2]) == 0 and (data[i - 3]) == 0)   or (BeginMarker == 1)): #if (((data[i - 2]) == 0 and (data[i - 3]) == 0 and (data[i - 4]) == 0) and (
                            #(data[i - 1]) == 19 or (data[i - 1]) == 14 or (data[i - 1]) == 15)) or (BeginMarker == 1):
                        if BeginMarker != 1:
                            MarkerNameNew1.append('  ')

                        MarkerNameNew1.append(string_Marker)

                        if (data[i + 1]) == 0 and (data[i + 2]) == 0:
                            BeginMarker = 0
                        else:
                            BeginMarker = 1

            MarkerNameNew2 = ''.join(str(x) for x in MarkerNameNew1)
            MarkerName = MarkerNameNew2.split(' ')
            i = 0
            while i in range(len(MarkerName)):

                if MarkerName[i] == '':
                    del MarkerName[i]
                i = i + 1

        if self.file_address.endswith(".mk2"):
            data_noinfo = 4
            data_length = 29

            for i in range(10):
                datastruct = '>dddIb'
                if i==0:
                    s = unpack(datastruct, data[data_noinfo:(29+data_noinfo)])
                else:
                    s = unpack(datastruct, data[(i * (30+9))+data_noinfo:(29 + (i * (30+9))+data_noinfo)])
                index[i]=s[0]
                Left[i] = s[1]
                Top[i] = s[2]
                Radius[i] = s[4]


            data_totallength=len(data)

            data_list=data[(29 + (10 * (30+9))+data_noinfo)+98:]

            string_Marker=data.decode('cp855',errors='ignore')

            MarkerName=string_Marker.split()

            MarkerNameNew1=[]
            BeginMarker=0
            start_byte=400
            for i in range(start_byte,len(data)):

                string_Marker=chr(data[i])
                #print(string_Marker)
                #print(data[i])
                if (str.isalpha(string_Marker) or string_Marker=='?' or string_Marker=='('  or string_Marker==')' ) and string_Marker!='ÿ' :
                            #print(data[i-1])
                            if (((data[i-2])==0 and (data[i-3])==0 and (data[i-4])==0 ))or (BeginMarker==1): #if (((data[i-2])==0 and (data[i-3])==0 and (data[i-4])==0 )and ((data[i-1])==19 or (data[i-1])==14))or (BeginMarker==1):
                                if BeginMarker!=1:
                                    MarkerNameNew1.append('  ')
                                #print (string_Marker)

                                MarkerNameNew1.append(string_Marker)
                                #print(data[i+1])
                                #print(MarkerNameNew1)
                                if (i+1 and i+2 )< len(data):
                                    if (data[i+1])==0 and (data[i+2])==0:
                                        BeginMarker=0
                                    else:
                                        BeginMarker= 1


            MarkerNameNew2=''.join(str(x) for x in MarkerNameNew1)
            MarkerName=MarkerNameNew2.split(' ')
            i=0
            while i in range (len(MarkerName)):

                if MarkerName[i]=='':
                    del MarkerName[i]
                i=i+1

        if self.file_address.endswith(".mk1"):
            MarkerName=MarkerName[1:], Left=Left[1:], Top= Top[1:], Radius=Radius[1:], index=index[1:]
        else:
            MarkerName=MarkerName[1:]

        return MarkerName,Left,Top,Radius, index
