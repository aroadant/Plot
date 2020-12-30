#!/usr/bin/env python
import os
import sys 
import subprocess as subp
import struct
import pickle
import numpy as np

from copy import deepcopy

#from scipy import interpolate


homed = os.getenv("HOME")
currd = os.getenv('PWD')

temed = "%s/Documents/tem/" % (homed)

NONE = 1e100

Gev_fm2 = 0.03894

def mt_n(option,temin,ofile):
  program = ["ls"]
  if option!="":
    program=option
    #proc = subp.Popen(program,stdin=subp.PIPE,stdout=subp.PIPE,stderr=subp.PIPE)
    if ofile=="":
      proc = subp.Popen(program,stdin=subp.PIPE,stdout=subp.PIPE)
    else:
      proc = subp.Popen(program,stdin=subp.PIPE,stdout=subp.PIPE,stderr=ofile)
  else:
    #proc = subp.Popen([program],stdin=subp.PIPE,stdout=subp.PIPE,stderr=subp.PIPE)
    proc = subp.Popen([program],stdin=subp.PIPE,stdout=subp.PIPE,stderr=ofile)

  if (os.path.isfile("control.txt")==1):
    if temin!="":
      a , b =proc.communicate(temin.encode("utf-8"))
    else:
      a , b =proc.communicate()
    return ([a,b])
  else:
    print("No file control.txt!")
    exit()

def mt(option,temin,ofile):
  program = ["ls"]
  if option!="":
    program=option
    #proc = subp.Popen(program,stdin=subp.PIPE,stdout=subp.PIPE,stderr=subp.PIPE)
    if ofile=="":
      proc = subp.Popen(program,stdin=subp.PIPE,stdout=subp.PIPE)
    else:
      proc = subp.Popen(program,stdin=subp.PIPE,stdout=subp.PIPE,stderr=ofile)
  else:
    #proc = subp.Popen([program],stdin=subp.PIPE,stdout=subp.PIPE,stderr=subp.PIPE)
    proc = subp.Popen([program],stdin=subp.PIPE,stdout=subp.PIPE,stderr=ofile)

  if temin!="":
    a , b =proc.communicate(temin.encode("utf-8"))
  else:
    a , b =proc.communicate()
  return ([a,b])


def get_path(name1):
  name1 = name1.split("/")
  name = name1[len(name1)-1]
  path = "/"
  for i in range(len(name1)):
    if i != len(name1)-1 :
      path = path + "/" + name1[i]
  path = path + "/"
  return ([path,name])

def pdf_joint(file_list,target_n = "",del_src=1):
  if target_n == "":
    target_n = "total.pdf"
  write_file = "-sOutputFile=%s" % (target_n)

  if (os.path.isfile(target_n) == 1):
    temstr = "/bin/mv " + target_n + (" %s/Documents/" % (homed))
    os.system(temstr)

  for file1 in file_list:
    if (os.path.isfile(file1) != 1):
      print(file1)
      print("Error in your existence!")
      #sys.exit()
      return -1
    
  optiono = ["gs","-dBATCH","-dNOPAUSE","-q","-sDEVICE=pdfwrite",write_file]
  optiono = optiono + file_list
  temstr = ""
  for tem1 in optiono:
    temstr = temstr + " "+ tem1
  os.system(temstr)

  if del_src == 1:
    for tems in file_list:
      #temstr = "/bin/mv " + tems + (" %s/Documents/" % (homed))
      temstr = "trash " + tems 
      os.system(temstr)
    #mt(optiono,"","")


def writetem(stri):
  fileo = open("tem.txt","a")
  fileo.write(stri)
  fileo.close()

def writelog(stri,file1="log.txt",mod="a"):
  fileo = open(file1,mod)
  fileo.write(stri)
  fileo.close()

def saveJ(Adic,fil="dat.pickle",mod = 1):
  if mod == 1:
    currentpath = temed
  else:
    currentpath = os.getcwd()

  if mod == 2:
    fil_n = fil
  else:
    pathX = "%s/X_dat" % (currentpath)
    if os.path.exists(pathX)!=True:
      fls = "mkdir %s/X_dat" % (currentpath)
      os.system(fls)
    file1 = open(pathX+"/"+fil,"wb+")
    file1.close()
    fil_n = pathX+"/"+fil

  #file1 = open(fil_n,"w+")
  file1 = open(fil_n,"wb+")
  ptems = pickle.dumps(Adic)
  file1.write(ptems)
  file1.close()

def readJ(fil="dat.pickle",mod=1,dtype=-1):
  if mod == 1:
    currentpath = temed
  else:
    currentpath = os.getcwd()

  if mod == 2:
    fil_n = fil
  else:
    pathX = "%s/X_dat" % (currentpath)
    fil_n = pathX+"/"+fil

  if (os.path.isfile(fil_n)==0):
    print("Do you really know him?")
    return 0
    #exit
  #fil = open(fil_n,"r")
  #Adic = pickle.load(fil)

  fil = open(fil_n,"rb")
  #latin1
  if dtype == -1:
    try:
      Adic = pickle.load(fil, encoding="bytes")
    except:
      Adic = pickle.load(fil)
  else:
    if dtype == '':
      Adic = pickle.load(fil)
    if dtype == 'str':
      Adic = pickle.load(fil, encoding="ascii")
    if dtype == 'np':
      Adic = pickle.load(fil, encoding="latin1")

  fil.close()
  return Adic


def stderror(lists):
  average1 = 0
  error1 = 0
  NN = len(lists)
  for dar in lists:
    average1 = average1 + float(dar)/NN
    error1 = error1 + float(dar)*float(dar)/(NN)
  resul = average1

  if NN == 1:
    error1 = 0
  else:
    try:
      error1 = np.sqrt(abs((error1-average1*average1)/(NN-1)))
    except:
      erro1 = 0
      print("Point error is wrong.")
  return ([resul,error1])

def read_f(filen):
  if os.path.isfile(filen)!=1:
    print("The file doesn't exist!")
    exit()

  temf = open(filen,"r")
  res = []
  for strtem in temf:
    rest = strtem.split("\n")[0]
    if (rest != ""):
      res.append(rest)
  return res 


def readf(fileo):
  if (os.path.isfile(fileo)):
    fil = open(fileo)
    restr = fil.read()
    fil.close()
    return restr
  else:
    print("File doesn't exits! What do you want!")
    exit()

#Get error from Jacknife data
#lists with the zero to be the average
def errorJack(lists,sqrr = 1,mode=0): 
  if len(lists) <= 3:
    return([lists[0],0.0])

  resul = 0 
  averageJ = 0 
  errorJ = 0 
  NN = len(lists)-1
  resul = NN*float(lists[0])+resul
  for dar in lists[1:]:
    averageJ = averageJ+float(dar)/NN
    errorJ = errorJ + float(dar)*float(dar)/NN
  if mode == 0:
    resul = resul-(NN-1)*averageJ
  if mode == 1:
    resul = averageJ
  if mode == 2:
    resul = float(lists[0])
  errorJ = errorJ - averageJ*averageJ
  try:
    if errorJ >=0:
      errortr = ((np.sqrt(NN-1))**sqrr) * np.sqrt(errorJ)
    else:
      errortr = 0
  except:
    errortr = 0 
  return([resul,errortr])
#-------------------------------------------------------------
def extend_line(xlist,ylist,elist,fac=0.1):
  if len(xlist) == 1:
    valuex = xlist[0]
    if valuex == 0:
      v0 = -fac
      v1 = fac
    else:
      v0 = valuex*(1-fac)
      v1 = valuex*(1+fac)

    valuey = ylist[0]
    valuee = elist[0]
    xlist = []
    ylist = []
    xlist.append(v0)
    xlist.append(v1)
    ylist.append(valuey)
    ylist.append(valuey)
    elist.append(valuee)
    elist.append(valuee)

  return [xlist,ylist,elist]


def in_range_num(num,t_list):
  flags = 1 
  if (num < t_list[0]):
    flags = 0 

  if (num > t_list[1]):
    flags = 0 

  if flags == 1:
    return 1
  else:
    return 0


def in_range(n_list,t_list):
  flags = 1 
  if (n_list[0] < t_list[0]):
    flags = 0 

  if (n_list[1] > t_list[1]):
    flags = 0 

  if flags == 1:
    return 1
  else:
    return 0

def get_size(file1):
  if (os.path.isfile(file1) == 1):
    optionto = ["ls","-all",file1]
    temstr1 = mt(optionto,"","")[0]
    temstr1 = temstr1.split()
    return int(temstr1[4])
  else:
    return -1

def get_vn(filen,sizet=-1,dtype=-1,off=0):
  if os.path.isfile(filen)!=1:
    print(filen)
    print("As the wind, empty heart and keep moving.")
    #Empty of heart and move as the wind!
    exit()

  redouble = []
  ###'>d' '<d' '>f' '<f'
  if isinstance(dtype,int) == 1:
      byt = 8
  else:
    ###default '<d'
    dt = np.dtype(dtype)
    byt = dt.itemsize
  #else:
  #  if dtype == 'float':
  #    byt = 8
  #  else:
  #    if dtype == 'float32':
  #      byt = 4
  #    else:
  #      print "Cannot understand your float size."
  #      sys.exit()
  #byt = 8
  if sizet < 0:
    sizet = get_size(filen)

    if (sizet != -1):
      ndouble = int(get_size(filen)/byt)
    else:
      return -1
  else:
    ndouble = sizet

  #if off == 0:
  #  offset = 0
  #else:
  offset = byt*off
  #orfile = open(filen,'rb')
  #for i in range(ndouble):
  #  tem = orfile.read(byt)
  #  redouble.append(float(struct.unpack('d',tem)[0]))

  #tem = orfile.read(byt*ndouble)
  #redouble = np.zeros(ndouble)
  #for i in range(ndouble):
  #  redouble[i] = float(struct.unpack('d',tem[i*8:(i*8+8)])[0])
  #redouble = np.fromfile(filen,dtype=dtype,count=ndouble)
  if isinstance(dtype,int) == 1:
    try:
      redouble = np.fromfile(filen,count=ndouble,offset=offset)
    except:
      redouble = np.fromfile(filen,count=ndouble)
    #redouble = np.fromfile(filen,count=ndouble)
  else:
    redouble = np.fromfile(filen,dtype=dt,count=ndouble,offset=offset)
  return redouble


#def get_vn(filen,sizet=-1,dtype=-1):
#  if os.path.isfile(filen)!=1:
#    print(filen)
#    print("As the wind, empty heart and keep moving.")
#    #Empty of heart and move as the wind!
#    exit()
#
#  redouble = []
#  ###'>d' '<d' '>f' '<f'
#  if isinstance(dtype,int) == 1:
#      byt = 8
#  else:
#    ###default '<d'
#    dt = np.dtype(dtype)
#    byt = dt.itemsize
#  #else:
#  #  if dtype == 'float':
#  #    byt = 8
#  #  else:
#  #    if dtype == 'float32':
#  #      byt = 4
#  #    else:
#  #      print "Cannot understand your float size."
#  #      sys.exit()
#  #byt = 8
#  if sizet < 0:
#    sizet = get_size(filen)
#    
#    if (sizet != -1):
#      ndouble = int(get_size(filen)/byt)
#    else:
#      return -1
#  else:
#    ndouble = sizet
#
#  #orfile = open(filen,'rb')
#  #for i in range(ndouble):
#  #  tem = orfile.read(byt)
#  #  redouble.append(float(struct.unpack('d',tem)[0]))
#
#  #tem = orfile.read(byt*ndouble)
#  #redouble = np.zeros(ndouble)
#  #for i in range(ndouble):
#  #  redouble[i] = float(struct.unpack('d',tem[i*8:(i*8+8)])[0])
#  #redouble = np.fromfile(filen,dtype=dtype,count=ndouble)
#  if isinstance(dtype,int) == 1:
#    redouble = np.fromfile(filen,count=ndouble)
#  else:
#    redouble = np.fromfile(filen,dtype=dt,count=ndouble)
#  return redouble


#def add_path(patho,l_file,follow = "",pre=""):
def add_path(patho,l_file,pre="",follow = ""):
  re_file = []
  for tem_str in l_file:
    re_file.append(patho + "%s%s%s" % (pre,tem_str,follow))
  return re_file
    
def simple_ava(xlist,ylist,elist,ava_range): 
  yava = 0.0
  eava = 0.0
  N_p = 0
  for x_i in range(len(xlist)):
    x_tem = xlist[x_i]
    if (x_tem >= ava_range[0]) and (x_tem <= ava_range[1]):
      N_p = N_p + 1
      yava = yava + ylist [x_i]
      eava = eava + elist [x_i]

  return [yava/N_p,eava/N_p]


def copy_v(n,num=[]):
  tem = []
  for i in range(n):
    tem.append(deepcopy(num))
  return tem

      
def shift_t(value_0,t_shi,fac):
  N_t = len(value_0)
  value_p = np.array([0.0]*(N_t+1))
  #copy_v(N_t+1,0.0)
  for t_i in range(N_t):
    value_p[t_i] = value_0[(t_i+t_shi)%N_t]

  if abs(fac[1])>1e-100:
    value_p[N_t] = fac[0]*value_p[0]/fac[1]
  else:
    value_p[N_t] = value_p[0]

  #value_p = np.array(value_p)
  #print value_p
  value_m = np.fliplr([value_p])[0]
  return (value_p*fac[0]+value_m*fac[1])[:N_t]


def write_vn(filen,vector,dtype=-1):
  #orfile = open(filen,'ab')
  #byt = 8
  #for tem_d in vector:
  #  orfile.write(struct.pack('d',tem_d))
  if dtype == -1:
    dtype = 'float'
  np.asarray(vector).astype(dtype).tofile(filen)
  return 1


class mom_3():
  def __init__(self,p_mom):
    self.Q_sq = 0.0
    self.p_mom = [0,0,0]
    self.s0 = 0
    self.s2 = 0
    self.s4 = 0

    if isinstance(p_mom,int):
      self.p_mom[2] = -50 + p_mom%(100)
      self.p_mom[1] = -50 + (int(p_mom/(100)))%100
      self.p_mom[0] = -50 + (int(p_mom/(10000)))%100
      self.p_mom = np.asarray(self.p_mom)

    if isinstance(p_mom,str):
      p_mom = int(p_mom)
      self.p_mom[2] = -50 + p_mom%(100)
      self.p_mom[1] = -50 + (int(p_mom/(100)))%100 
      self.p_mom[0] = -50 + (int(p_mom/(10000)))%100 
      self.p_mom = np.asarray(self.p_mom)

    if isinstance(p_mom,list):
      if len(p_mom) == 3:
        self.p_mom = p_mom
        self.p_mom = np.asarray(self.p_mom)
      else:
        print("Your heart will break.")
        sys.exit()

    if isinstance(p_mom,float):
      self.Q_sq = p_mom

    for i in range(3):
      self.s0 = self.s0 + int(np.abs(self.p_mom[i]**1))
      self.s2 = self.s2 + int(np.abs(self.p_mom[i]**2))
      self.s4 = self.s4 + int(np.abs(self.p_mom[i]**4))

  def mom_sq(self,L_i=-1):
    fac = 1.0
    if L_i == -1:
      fac = 1.0
    else:
      fac = (2*np.pi)/L_i

    Q_sq = 0.0
    for q_tem in self.p_mom:
      #Q_sq = Q_sq + 4*np.sin(q_tem*fac/2)**2
      Q_sq = Q_sq + (q_tem*fac)**2
    self.Q_sq = Q_sq
    return self.Q_sq


  def add_mom(self,p_0):
    return np.asarray(p_0.p_mom)+ np.asarray(self.p_mom)

  def mom_int(self):
    v = 0
    v = v + int((50 + self.p_mom[0])*10000)
    v = v + int((50 + self.p_mom[1])*100)
    v = v + int((50 + self.p_mom[2])*1)
    #return int(self.p_mom[0]*10000)+int(self.p_mom[1]*100)+int(self.p_mom[2])
    return v

  def equalmom(self,mom1):
    if self.s0 == mom1.s0 and  self.s2 == mom1.s2  and self.s4 == mom1.s4:
      return 1
    else:
      return 0
  

def anti_Jack_l(E_list,mod=0):
  #print E_list[0]
  x_sum = 0
  if mod == 0:
    N_0 = len(E_list)
    for i in range(N_0):
      x_sum = x_sum + E_list[i]

  if mod == 1:
    N_0 = len(E_list)-1
    x_sum = E_list[0]*N_0

  if mod == 2:
    N_0 = len(E_list) - 1
    for i in range(N_0):
      x_sum = x_sum + E_list[i+1]
    #x_sum = E_list[0]*N_0


  re_Elist = list(range(N_0))
  for i in range(N_0):
    if mod == 0 or mod == 1:
      re_Elist[i] = x_sum - (N_0-1)*E_list[i+mod]
    if mod == 2:
      re_Elist[i] = x_sum - (N_0-1)*E_list[i+1]

  return re_Elist

def del_range(t_list,ava_range):
  res = []
  for t in t_list:
    if t>=ava_range[0] and t<=ava_range[1]:
      res.append(t)
  return res


#tem = mom_3(np.asarray([0, 1, 2]))
#tem = mom_3(10)
#print tem.p_mom
#print tem.mom_int()
#print tem.mom_sq(32)

def Jack_data(value_l):
  re = copy_v(len(value_l)+1,0.0)
  re[0] = np.average(value_l)
  for c_o in range(len(value_l)):
    tem_list = np.delete(value_l,c_o)
    re[c_o+1] = np.average(tem_list)
  return re

def fake_gauss(mu,sigma,N_n):
  values_Ja = np.random.normal(mu, sigma/np.sqrt(N_n), N_n)
  values_or = anti_Jack_l(values_Ja)
  return [values_or,values_Ja]


#Jacknife the configuration for computation and store in dic_J
#create a dictionary with "-1" to be the whole average
def Jac_confs(c_dic,c_list,t_list):
  dic_J = {}
  for t_tem in t_list:
    value_J = []
    value_list = []
    for c_tem in c_list:
      value_list.append(c_dic[(c_tem,t_tem)])
    dic_J[("-1",t_tem)] = np.average(value_list)
    for c_o in range(len(value_list)):
      tem_list = np.delete(value_list,c_o)
      dic_J[(c_list[c_o],t_tem)] = np.average(tem_list)
  c_J_list = deepcopy(["-1"] + c_list)
  return dic_J

def anti_Jack_confs(c_dic,c_list,t_list,mod=0):
  dic_J = {}
  for t_tem in t_list:
    value_J = []
    value_list = []
    for c_tem in c_list:
      value_list.append(c_dic[(c_tem,t_tem)])
    value_list = anti_Jack_l(value_list,mod)
    for c_o in range(len(value_list)):
      dic_J[(c_list[c_o+mod],t_tem)] = value_list[c_o]
  return dic_J

def error_confs(c_dic,c_list,t_list):
  error = {}
  for t_tem in t_list:
    value_J = []
    value_list = []
    for c_tem in c_list:
      value_list.append(c_dic[(c_tem,t_tem)])
    error[(t_tem)] = stderror(value_list)
  return error


def get_clust(values,diff):
  sortx = np.argsort(values)
  clust = []

  tem =  []
  cur = values[sortx[0]]
  for i in range(len(values)):
    if abs(values[sortx[i]]-cur) <= diff:
      tem.append(sortx[i])
    else:
      clust.append(deepcopy(tem))
      tem = []
    cur = values[sortx[i]]
  clust.append(tem)
  return clust

def get_full(dat,diffx,diffy):
  full_clust = []

  radiusx = copy_v(len(dat),0.0)
  for i in range(len(dat)):
    radiusx[i] = dat[i][0]

  clustx = get_clust(radiusx,diffx)
  #print len(clustx)
  for c_i in range(len(clustx)):
    radiusy = []
    for i in range(len(clustx[c_i])):
      radiusy.append(dat[clustx[c_i][i]][1])

    #print len(radiusy)
    clust_temy = get_clust(radiusy,diffy)
    for cy in range(len(clust_temy)):
      num_tem = []
      for cyi in range(len(clust_temy[cy])):
        cur_i = clustx[c_i][clust_temy[cy][cyi]]
        num_tem.append(dat[cur_i])
      full_clust.append(deepcopy(num_tem))

  return full_clust

def get_error(cus_i,mod=0):
  xvalue = []
  yvalue = []
  for i in range(len(cus_i)):
    tem = cus_i[i]
    xvalue.append(tem[0])
    yvalue.append(tem[1])

  x   = (max(xvalue)+min(xvalue))/2.0
  dx  = (max(xvalue)-min(xvalue))/2.0
  y   = (max(yvalue)+min(yvalue))/2.0
  dy  = (max(yvalue)-min(yvalue))/2.0
  if mod == 0:
    return [x,dx,y,dy]
  if mod == 1:
    xi = np.argsort(xvalue)
    #use = [yvalue[xi[0]],yvalue[xi[1]],yvalue[xi[3]],yvalue[xi[4]],yvalue[xi[5]],yvalue[xi[6]],yvalue[xi[7]],yvalue[xi[8]]]
    use = [yvalue[xi[0]],yvalue[xi[1]],yvalue[xi[3]],yvalue[xi[4]]]
    #y   = (max(use)+min(use))/2.0
    dy  = (max(use)-min(use))/2.0

    return [x,dx,y,dy]
    

def read_plot(dat,diffx=0.05,diffy=0.1,mod=0):
  clust = get_full(dat,diffx,diffy)
  x_list = []
  y_list = []
  e_list = []

  for i in range(len(clust)):
    tem = get_error(clust[i],mod=mod)
    x_list.append(tem[0])
    y_list.append(tem[2])
    e_list.append(tem[3])
  return [x_list,y_list,e_list,clust]


def get_dis(values,step=0.2):
  tem = stderror(values)
  Ndat = len(values)
  aver = tem[0]
  var = np.sqrt(Ndat)*tem[1]

  #print tem
  #print var

  minv = min(values)
  maxv = max(values)
  dvar = var*step
  Num = int((maxv-minv)/dvar)+2
  countL = np.zeros(Num)
  valueL = np.zeros(Num)
  #minv-dvar
  #maxv+dvar
  for ci in range(Num):
    curmin = minv + dvar*ci
    curmax = minv + dvar*ci + dvar
    #valueL[ci] = (curmin + curmax)/2.0
    valueL[ci] = ((curmin + curmax)/2.0-aver)/var
    for vi in values:
      if vi>curmin and vi< curmax:
        countL[ci] = countL[ci] + 1

  valueL0 = []
  countL0 = []
  for i in range(len(valueL)):
    if countL[i] != 0.0:
      valueL0.append(valueL[i])
      countL0.append(countL[i])

  return [valueL0,countL0]


def get_mean(values,step=0.3):
  Ndat = len(values)
  mean = 0.0
  curcount = 0

  tem = stderror(values)
  aver = tem[0]
  stdv = tem[1]
  [valueL,countL] = get_dis(values,step)
  for i in range(len(valueL)):
    if countL[i] > curcount:
      mean = (valueL[i]*np.sqrt(Ndat)*stdv)+aver
      curcount = countL[i]

  countdiff = []
  for i in range(len(valueL)):
    countdiff.append(abs(curcount/2.0 - countL[i]))

  diffhalfcount = min(countdiff)
  for i in range(len(valueL)):
    if abs(curcount/2.0 - countL[i]) == diffhalfcount:
      stdvcount = (valueL[i]*np.sqrt(Ndat)*stdv)+aver
  return [mean,stdvcount]

def get_mass_print(fil):
  a = np.loadtxt(fil)
  for i in range(len(a)):
    print("%8.6f"%a[i])
  
def perm_parity(lst,mod=0):
  '''\
  Given a permutation of the digits 0..N in order as a list,
  returns its parity (or sign): +1 for even parity; -1 for odd.
  '''

  if mod == 1:
    lst = list(np.asarray(lst)-1)

  parity = 1
  for i in range(0,len(lst)-1):
    pos = lst.index(i)
    if pos%2 == 1:
      parity *= -1
    lst.remove(i)
    #if lst[i] != i:
    #  parity *= -1
    #  mn = min(range(i,len(lst)), key=lst.__getitem__)
    #  lst[i],lst[mn] = lst[mn],lst[i]
  return parity


def sort_list(a):
  if len(a) == 0:
    print("sort list is zero.")
    sys.exit()
  if isinstance(a[0],list):
    Nlist = len(a)
    if Nlist == 1:
      return numpy.asarray([np.sort(a[0])])
    for ni in range(Nlist):
      if len(a[ni]) != len(a[0]):
        print("list dimension wrong!!!")
        sys.exit()

    nL = np.argsort(a[0])
    b = copy_v(len(a),[])
    for ni in range(Nlist):
      for k in range(len(nL)):
        b[ni].append(a[ni][nL[k]])
    return np.asarray(b)
  else:
    return np.sort(a)

def interfit(alist,interp=0):
  if interp == 0:
    return alist

  if interp == 1:
    interp = 0.1

  Ni = len(alist)
  if Ni == 0 or Ni == 1:
    return alist

  blist = copy_v(3,[])
  fs = []
  for ni in range(Ni-1):
    fs.append(interpolate.interp1d(np.asarray(alist[0]), np.asarray(alist[ni+1])))
  #fs.append(interpolate.interp1d(np.asarray(alist[0]), np.asarray(alist[len(alist)-1])))

  fxlist = alist[0]
  Lx = max(fxlist)-min(fxlist) + 0.0
  nx = len(fxlist)*1.0
  for xi in np.arange(min(fxlist),max(fxlist),interp*Lx/(nx)):
    blist[0].append(xi)
    for ni in range(Ni-1):
      blist[ni+1].append(fs[ni](xi))

  return deepcopy(np.asarray(blist))

#li = [iprj,idir,maplist[mi]]
#li = list(np.asarray(li)-1)

def geterror(v,mode = 1):
  import gvar
  if mode == 0:
    return gvar.gvar(v)
  tem = v.split("(")
  t0 = []
  for l in tem:
    t0.append(l.split(")")[0])

  if len(t0) <2:
    print("Error for %s"%t0)
    sys.exit()

  use = gvar.gvar(v)
  e0  = use.sdev

  if len(t0) == 2:
    return gvar.gvar(v)

  if len(t0) >= 3:
    mean = t0[0]
    err = 0.0
    for i in range(len(t0)-1):
      err = err + float(t0[i+1])**2
    err = np.sqrt(err)
    err = int(np.round(err,0))
    return gvar.gvar("%s(%s)"%(mean,err))


from solver_gen import get_eff_m3

def get_mass_log(c_dic,c_list,t_list,flags = 0,source=64):
  dic_J = Jac_confs(c_dic,c_list,t_list)
  c_listo = ["-1"] + c_list

  xlist = []
  ylist = []
  elist = []

  for t_tem in t_list:
    flag = 1
    xlist.append(t_tem)

    value_list = []
    for c_tem in c_listo:
      v1 = dic_J[(c_tem,t_tem)]
      if v1 != 0:
        if t_tem - 1 < min(t_list):
          v0 = dic_J[(c_tem,t_tem)]
        else:
          v0 = dic_J[(c_tem,t_tem-1)]

        if t_tem + 1 > max(t_list):
          v2 = dic_J[(c_tem,t_tem)]
        else:
          v2 = dic_J[(c_tem,t_tem)]

        gm = get_eff_m3([v0,v1,v2],flag = flags,list_t=[source/2.0,t_tem],res=0.3)
        if gm[1] == 0:
          tem = gm[0]
        if gm[1] == 1:
          flag = 0
          tem = 0
          break

      else:
        flag = 0
        tem = 0
        break
      value_list.append(tem)

    if flag == 1:
      re = errorJack(value_list)

      ylist.append(re[0])
      elist.append(re[1])

    if flag == 0:
      ylist.append(0.0)
      elist.append(0.0)

  return deepcopy([xlist,ylist,elist])
  #errorJack

typelist = ["other", "x", "y", "z", "t", "d", "c", "d2", "c2", "complex", "mass", "smear", "displacement", "s_01", "s_02", "s_03", "s_11", "s_12", "s_13", "d_01", "d_02", "d_03", "d_11", "d_12", "d_13", "conf", "operator", "momentum", "direction", "t2", "mass2", "column", "row", "temporary", "temporary2", "temporary3", "temporary4", "errorbar", "operator2", "param", "fit_left", "fit_right", "jackknife", "jackknife2", "jackknife3", "jackknife4", "summary", "channel", "channel2", "eigen", "d_row", "d_col", "c_row", "c_col", "parity", "noise", "evenodd", "disp_x", "disp_y", "disp_z", "disp_t", "t3", "t4", "t_source", "t_current", "t_sink", "nothing", "function", "datapoint", "temporary1", "temporary5", "temporary6", "temporary7", "temporary8", "temporary9", "bootstrap", "xi_g", "xi_g2", "xi_f", "xi_f2", "beta", "nx", "ny", "nz", "nt", "operator_source", "operator_sink", "operator_current", "series"]

def read_dim(filen):
  if os.path.isfile(filen)!=1:
    print(filen)
    print("As the wind, empty heart and keep moving.")
    #Empty of heart and move as the wind!
    exit()

  orfile = open(filen,'rb')
  #tem = orfile.read(12800*8)
  #Li = struct.iter_unpack('i',tem)
  #print(Li)
  Li = []
  for i in range(int(12800*8/4)):
    #print(ltem)
    tem = orfile.read(4)
    a = struct.unpack('i',tem)
    #Li.append(a[0])
    Li.append(a[0])

  Ldim = Li[0]
  off = 1

  dim_name = []

  def get_v(Li,off):
    name = typelist[Li[off]]
    di = Li[off+1]
    return [name,di]

  for idim in range(Ldim):
    off = 1 + 1026*idim
    [name,di] = get_v(Li, off)
    tem = []
    for i in range(di):
      tem.append(Li[off + 2 + i])
    dim_name.append([name,di,deepcopy(tem)])
    #print([name,di])

  return(deepcopy(dim_name))


