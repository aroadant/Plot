import os
import sys
import numpy as np
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import datetime
from matplotlib.backends.backend_pdf import PdfPages

#from general_fun import *
###TEST
###TEST

homed = os.getenv("HOME")
currd = os.getenv('PWD')
currentpath = os.getcwd()

sys.path.append("%s/" % (currd))
from general_fun import saveJ
from general_fun import readJ
from general_fun import pdf_joint
from general_fun import NONE
from general_fun import in_range_num

from general_fun import stderror

def get_v(custom_v,c_len):
  #isinstance(["1","2"], (list))
  if isinstance(custom_v,(int)):
    if custom_v == -1:
      if c_len[1] == len(c_len[0])-1:
        tem_n = c_len[1]
        c_len[1] = 0
      else:
        tem_n = c_len[1]
        c_len[1] = c_len[1] + 1
      return c_len[0][tem_n]
    else:
      #c_len[0].remove(c_len[0][custom_v])
      return c_len[0][custom_v]

  if isinstance(custom_v,(str)):
    if custom_v == "":
      return c_len[0][0]
    else:
      return custom_v

class c_plot:
  def __init__(self,rangex=[],rangey=[],xlab="xlab",ylab="ylab",lag="",title="Plot Sample",legen_s=0,sav="tem",tp="png",showd=0,pdf_num=0):
    #plt.close()
    #plt.cla()
    #plt.clf()

    pathfig = "%s/pdf" % (currentpath)
    if os.path.exists(pathfig)!=True:
      fls = "mkdir %s/pdf" % (currentpath)
      os.system(fls)

    self.rangex = rangex
    self.rangey = rangey
    self.lag = lag
    self.xlab = xlab
    self.ylab = ylab
    self.title = title
    self.sav = sav
    self.legen_s = legen_s
    self.tp = tp
    self.showd = showd

    #You can set the grid_s as True
    self.grid_s = False
    self.log_x = 0

    #color settings
    cor = ['b','g','r','c','k','m','y','grey','purple','coral','chartreuse','violet','maroon','cadetblue','cyan']
    cor = cor + cor + cor 
    self.cmap = plt.get_cmap('gnuplot')
    self.cor = cor + [self.cmap(i) for i in np.linspace(0, 1, 10)]
    self.cor = [cor,0]
    #temrinal color at 16
    #cmap(np.random.random())
    #used cycle of colors
    #line settings
    lin = ["None","-","-.","--"," ",":",""]
    lin = lin + lin + lin
    self.lin = [lin,0]

    #marker settings
    #mark None to be ""
    mark = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","*","h","H","+","x","D","d","|","_"]
    mark = mark + mark + mark
    self.mark = [mark,0]
    #terminal at len(mark[0])
    label_l = []
    for i in range(0,100,1):
      label_l.append(str(i))
    self.label_l = [label_l,0]

    #Variable for pdf_pages
    self.pdf_num = pdf_num
    self.Jac_dat = []
    self.name = ""
    self.legen_size = 8

    self.alpha = 0.5


  #Jac mark 0
  def a_line(self,xlist,ylist,lin='-',cor= -1,mark="",label_l= -1,mks=-1):
    #iteration over the default options
    xlist = np.asarray(xlist)
    ylist = np.asarray(ylist)

    lin = get_v(lin,self.lin)
    mark = get_v(mark,self.mark)
    cor = get_v(cor,self.cor)
    label_l = get_v(label_l,self.label_l)
    #print lin
    #print mark
    #print cor
    #print label_l

    #Get the instance lines of instance
    #line = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,label=label_l)[0]
    #Or the alternate grammar to use the a, = [12] to be a = 12
    #line, = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,label=label_l)
    tem = [xlist,ylist,lin,cor,mark,label_l,mks]

    #Jac_name = "plot_tem_%s_%s_%s" % (self.name,self.pdf_num,len(self.Jac_dat))
    #saveJ(tem,Jac_name)
    #self.Jac_dat.append([0,Jac_name])

    self.Jac_dat.append([0,deepcopy(tem)])
    return [lin,cor,mark,label_l,mks]

    #plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,markersize=mks,label=label_l)

  #Jac mark 1
  def err_line(self,xlist,ylist,bary,lin='-',cor= -1,mark="",label_l= -1,mks=-1):
    xlist = np.asarray(xlist)
    ylist = np.asarray(ylist)
    bary = np.asarray(bary)

    lin = get_v(lin,self.lin)
    mark = get_v(mark,self.mark)
    cor = get_v(cor,self.cor)
    #print label_l
    label_l = get_v(label_l,self.label_l)
    #line,_,_ = plt.errorbar(xlist,ylist,yerr=bary,ls=lin,color=cor,marker=mark,label=label_l)
    #print label_l
    #line = plt.errorbar(xlist,ylist,yerr=bary,ls=lin,color=cor,marker=mark,label=label_l)[0]
    ##plt.errorbar(xlist,ylist,yerr=bary,ls=lin,color=cor,marker=mark,label=label_l)
    #if mks != -1:
    #  line.set_markersize(mks)
    #plt.fill_between([0, 1], [0, 1])

    tem = [xlist,ylist,bary,lin,cor,mark,label_l,mks]

    #Jac_name = "plot_tem_%s_%s" % (self.pdf_num,len(self.Jac_dat))
    #Jac_name = "plot_tem_%s_%s_%s" % (self.name,self.pdf_num,len(self.Jac_dat))
    #saveJ(tem,Jac_name)
    #self.Jac_dat.append([1,Jac_name])
    self.Jac_dat.append([1,deepcopy(tem)])
    return [lin,cor,mark,label_l,mks]
    
  def a_line_twoy(self,xlist,ylist,lin='-',cor= -1,mark="",label_l= -1,mks=-1):
    #iteration over the default options
    xlist = np.asarray(xlist)
    ylist = np.asarray(ylist)

    lin = get_v(lin,self.lin)
    mark = get_v(mark,self.mark)
    cor = get_v(cor,self.cor)
    label_l = get_v(label_l,self.label_l)
    #print lin
    #print mark
    #print cor
    #print label_l

    #Get the instance lines of instance
    #line = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,label=label_l)[0]
    #Or the alternate grammar to use the a, = [12] to be a = 12
    #line, = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,label=label_l)
    tem = [xlist,ylist,lin,cor,mark,label_l,mks]

    #Jac_name = "plot_tem_%s_%s_%s" % (self.name,self.pdf_num,len(self.Jac_dat))
    #saveJ(tem,Jac_name)
    #self.Jac_dat.append([0,Jac_name])

    self.Jac_dat.append([0+10,deepcopy(tem)])
    return [lin,cor,mark,label_l,mks]

  #Jac mark 2
  def err_bound(self,xlist,ylist,bary,lin='-',cor= -1,mark="",label_l= -1,mks=-1):
    lin = get_v(lin,self.lin)
    mark = get_v(mark,self.mark)
    label_l = get_v(label_l,self.label_l)
    xlist = np.asarray(xlist)
    ylist = np.asarray(ylist)
    bary = np.asarray(bary)

    cor = get_v(cor,self.cor)
    #line, = plt.fill_between(xlist, ylist-bary, ylist+bary,color=cor)
    #No return variables
    ###plt.fill_between(xlist, ylist-bary, ylist+bary,color=cor,label=label_l)

    #if mks == -1:
    #  plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,label=label_l)
    #  plt.fill_between(xlist, ylist-bary, ylist+bary)
    #else:
    #  plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,markersize=mks,label=label_l)
    #  pl.fill_between(xlist, ylist-bary, ylist+bary)

    tem = [xlist,ylist,bary,cor,label_l]
    #Jac_name = "plot_tem_%s" % (len(self.Jac_dat))
    #Jac_name = "plot_tem_%s_%s" % (pdf_num,len(self.Jac_dat))
    #Jac_name = "plot_tem_%s_%s_%s" % (self.name,self.pdf_num,len(self.Jac_dat))
    #saveJ(tem,Jac_name)
    #self.Jac_dat.append([2,Jac_name])
    self.Jac_dat.append([2,deepcopy(tem)])
    return [cor,label_l]

  #add_text(10, 720.000, r'$\mu=100,\ \sigma=15$')
  def add_text(self,pos_x,pos_y,tem_str,fontsize=-1):
    tem = [pos_x,pos_y,tem_str,fontsize]
    self.Jac_dat.append([3,deepcopy(tem)])
    return 0.1

  def sav_J(self):
    self.name_i="Plot_tem_%s"%(self.name)
    saveJ(self.Jac_dat,self.name_i)

  def red_J(self):
    self.Jac_dat = readJ(self.name_i)

  def check_list(self,xlist,ylist,bary=-1):
    #if bary == None:
    #  bary = range(len(ylist))
    if isinstance(bary,int)==1:
      bary = list(range(len(ylist)))

    del_i=[]
    for y_i in range(len(ylist)):
      flag = 0
      if self.lag=="log" and ylist[y_i] <= 1e-100:
        flag = 1
      if abs(ylist[y_i]) >= NONE/2.0:
        flag = 1
      if self.rangex !=[]:
        flag = 1-in_range_num(xlist[y_i],self.rangex)
        #print flag

      if flag == 1:
        del_i.append(y_i)
        self.del_list.append([y_i,ylist[y_i],xlist[y_i]])

    xlist = np.delete(xlist,del_i)
    ylist = np.delete(ylist,del_i)
    bary  = np.delete(bary,del_i)

    return [xlist,ylist,bary]

  def sav_dat(self,names):
    saveJ(self.Jac_dat,names)

  def add_Jac(self,datJ):
    for i in range(len(datJ)):
      self.Jac_dat.append(datJ[i])

  def plot_f(self):
    plt.close()
    plt.cla()
    plt.clf()

    if self.pdf_num != 0:
      fig_0 = plt.figure()

    #set up the initial to get the initial condition
    #self.bary = []
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    self.del_list = []
    for Jac_i in self.Jac_dat:
      if Jac_i[0] == 0:
        #[xlist,ylist,lin,cor,mark,label_l,mks] = readJ(Jac_i[1])
        [xlist,ylist,lin,cor,mark,label_l,mks] = Jac_i[1]
        [xlist,ylist,tem] = self.check_list(xlist,ylist)

        if len(xlist) > 0:
          if label_l == 0.1:
            line, = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark)
          else:
            line, = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,label=label_l)

          if mks != -1:
            line.set_markersize(mks)
            line.set_linsize(mks)
        else:
          line, = plt.plot([1.0],[1.0],ls=lin,color=cor,marker=mark,label="WRONG!")


      if Jac_i[0] == 1:
        #[xlist,ylist,bary,lin,cor,mark,label_l,mks] = readJ(Jac_i[1])
        [xlist,ylist,bary,lin,cor,mark,label_l,mks] = Jac_i[1]
        [xlist,ylist,bary] = self.check_list(xlist,ylist,bary)
        
        if len(xlist) > 0:
          if label_l == 0.1:
            line = plt.errorbar(xlist,ylist,yerr=bary,ls=lin,color=cor,marker=mark)[0]
          else:
            line = plt.errorbar(xlist,ylist,yerr=bary,ls=lin,color=cor,marker=mark,label=label_l)[0]
          if mks != -1:
            line.set_markersize(mks)
        else:
          line = plt.errorbar([1.0],[1.0],[1.0],ls=lin,color=cor,marker=mark,label="WRONG!")[0]
          

      if Jac_i[0] == 2:
        #[xlist,bary,bary,cor,label_l] = readJ(Jac_i[1])
        [xlist,ylist,bary,cor,label_l] = Jac_i[1]
        [xlist,ylist,bary] = self.check_list(xlist,ylist,bary)
        #[xlist,ylist,bary,lin,cor,mark,label_l,mks] = readJ(Jac_i[1])
        flag = 1
        for tem in bary:
          if abs(tem) < 1e-100:
            #print "tem!"
            flag = 0
        if len(xlist) > 0 and flag:
          if label_l == 0.1:
            plt.fill_between(xlist, ylist-bary, ylist+bary,facecolor=cor, alpha=self.alpha)
          else:
            plt.fill_between(xlist, ylist-bary, ylist+bary,facecolor=cor,label=label_l, alpha=self.alpha)


      if Jac_i[0] == 3:
        [pos_x,pos_y,tem_str,fontsize] =  Jac_i[1]
        #[xlist,bary,bary,cor,label_l] = readJ(Jac_i[1])
        #[xlist,ylist,bary,lin,cor,mark,label_l,mks] = readJ(Jac_i[1])
        if fontsize == -1:
          plt.text(pos_x,pos_y,tem_str)
        else:
          plt.text(pos_x,pos_y,tem_str,fontsize=fontsize)


    #print "DELETE LIST!"
    #print self.del_list

    if self.legen_s == 1:
      plt.legend(loc='best')
    if self.legen_s == 2:
      plt.legend(loc='best',prop={'size':self.legen_size})


    #if self.legen_s == -1:
    #  plt.legend(handler_map={li: HandlerLine2D(numpoints=4)})

    xlab = r"%s" % (self.xlab)
    ylab = r"%s" % (self.ylab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(self.title)

    plt.grid(self.grid_s)

    
    ####self.ax0 = plt.axes()
    ###ax0=plt.axes()
    if self.rangex != []:
      plt.axes().set_xlim(self.rangex[0],self.rangex[1])
    if self.rangey != []:
      plt.axes().set_ylim(self.rangey[0],self.rangey[1])
    
    #ax0.set_xticks(np.arange(0,10,1)) 
    ##self.ax0 = ax0
    ##self.plt = plt

    for Jac_i in self.Jac_dat:
      if Jac_i[0] == 10:
        [xlist,ylist,lin,cor,mark,label_l,mks] = Jac_i[1]
        [xlist,ylist,tem] = self.check_list(xlist,ylist)
        if len(xlist) > 0:
          ax2 = plt.axes().twinx()
          ax2.plot(xlist, ylist, cor)
          #ax2.set_ylabel('sin', color=cor)
          #ax2.tick_params('y', colors=cor)
          #if label_l == 0.1:
          #  line, = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark)
          #else:
          #  line, = plt.plot(xlist,ylist,ls=lin,color=cor,marker=mark,label=label_l)

          #if mks != -1:
          #  line.set_markersize(mks)
          #  line.set_linsize(mks)
        else:
          line, = plt.plot([1.0],[1.0],ls=lin,color=cor,marker=mark,label="WRONG!")


    #set up the log plot
    if self.lag=="log":
      #ax0.set_xscale('log')
      plt.axes().set_yscale('log')

    if self.log_x==1:
      plt.axes().set_xscale('log')

    #plt.tight_layout()
    plt.title(self.title)

    
    if self.sav == "":
      self.sav_f = ("%s/pdf/%s.%s" % (currentpath,self.title,self.tp))
    else:
      self.sav_f = ("%s/pdf/%s.%s" % (currentpath,self.sav,self.tp))

    if self.pdf_num == 0:
      plt.savefig(self.sav_f,bbox_inches='tight')

      if self.showd == 0:
        #print "What?"
        #fig_0.show()
        plt.show()
        #plt.cla()
        #plt.clf()
      else:
        plt.cla()
        plt.clf()

      #plt.close()
    else:
      #plt.close()
      #return plt.figure()
      return fig_0


# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
#with PdfPages('multipage_pdf.pdf') as pdf:

class pdf_page():

  def __init__(self,title="test",name=""):
    self.title = title
    self.fig_l = []
    self.name = name

  def add_p(self,plot_i):
    plot_i.pdf_num = len(self.fig_l) + 1
    self.fig_l.append(deepcopy(plot_i))
    #self.fig_l[len(self.fig_l)-1].plot_i.pdf_num = len(self.fig_l) + 1
    #plot_i.name = self.name

  def plot(self):

    pdf = PdfPages(self.title)
    fig_l = []
    for plot_i in self.fig_l:
      fig_l.append(plot_i.plot_f())

    for fig_i in fig_l:
      pdf.savefig(fig_i)
      #pdf.savefig()
      #pdf.savefig(fig=plt.figure())
      plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = 'Gen Wang'
    #d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    #d['Keywords'] = 'PdfPages multipage keywords author title subject'
    #d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()

    pdf.close()


class pdf_j():
  def __init__(self,name=""):
    if name =="":
      self.name = "test.pdf"
    else:
      self.name = name

    self.file_list = []

  def add_p(self,name_i,mod=1):
    if mod == 1:
      self.file_list.append("%s/pdf/%s.pdf"%(currd,name_i))
    else:
      self.file_list.append("%s/%s"%(currd,name_i))

  def plot(self):
    if len(self.file_list)>0:
      pdf_joint(self.file_list,"%s/pdf/%s.pdf"%(currd,self.name))






def plot_cor(c_dic_l,nameL,c_listL,t_listL,save="test",title="",rangex=[],rangey=[],showd=0,afac=1.0):
  pi_plot = c_plot(rangex=rangex,rangey=rangey,xlab="time",ylab="Corr",title=title,sav=save,tp="pdf",showd=showd,lag="")
  pi_plot.grid_s  = True
  pi_plot.legen_s = 2
  pi_plot.legen_size = 10

  for cLi in range(len(c_dic_l)):
    xlist = []
    ylist = []
    elist = []
    for t in t_listL[cLi]:
      xlist.append(t*afac+cLi*0.013)
      #xlist.append(t*afac)
      #xlist.append(t)
      value = []
      for c in c_listL[cLi]:
        value.append(1.0*c_dic_l[cLi][(c,t)])
      tem = stderror(value)
      ylist.append(tem[0])
      elist.append(tem[1])

    pi_plot.err_line(xlist,ylist,elist,lin="",label_l=r"%s"%(nameL[cLi]))

  pi_plot.plot_f()

def plot_err(c_dic_l,nameL,c_listL,t_listL,save="test",title="",rangex=[],rangey=[],showd=0,afac=1.0):
  pi_plot = c_plot(rangex=rangex,rangey=rangey,xlab="time",ylab=r"$Error^2$",title=title,sav=save,tp="pdf",showd=showd,lag="")
  pi_plot.grid_s  = True
  pi_plot.legen_s = 2
  pi_plot.legen_size = 10

  for cLi in range(len(c_dic_l)):
    xlist = []
    ylist = []
    elist = []
    eelist = []
    for t in t_listL[cLi]:
      xlist.append(t*afac+cLi*0.013)
      #xlist.append(t)
      value = []
      for c in c_listL[cLi]:
        value.append(1.0*c_dic_l[cLi][(c,t)])
      tem = stderror(value)
      #ylist.append(tem[0])
      #elist.append(tem[1])

      #eelist.append(tem[0]/tem[1])
      eelist.append(tem[1]**2)

    pi_plot.a_line(xlist,eelist,lin="",label_l=r"$E^2$ %s"%(nameL[cLi]))

  pi_plot.plot_f()


def plot_ratio_err(c_dic_l,nameL,c_listL,t_listL,save,simple=1,title="",inverse=0,rangex=[],rangey=[],flagsn=0,showd=0,afac=1.0):
  pi_plot = c_plot(rangex=rangex,rangey=rangey,xlab="time",ylab=r"$Error^2$ Ratio",title=title,sav=save,tp="pdf",showd=showd,lag="")
  pi_plot.grid_s  = True
  pi_plot.legen_s = 2
  pi_plot.legen_size = 10

  #flagsn = 0


  ee0 = []
  valueL0 = []
  for t in t_listL[0]:
    value = []
    for c in c_listL[0]:
      value.append(1.0*c_dic_l[0][(c,t)])
    valueL0.append(deepcopy(value))

    tem = stderror(value)
    if flagsn == 0:
      ee0.append(tem[1]**2)
    if flagsn == 1:
      ee0.append((tem[1]/tem[0])**2)

  for cLi in range(len(c_dic_l)):
    xlist = []
    eelist = []
    valueL1 = []
    for t in t_listL[cLi]:
      if rangex != []:
        if t >= rangex[1]:
          continue

      xlist.append(t*afac+cLi*0.013)
      #xlist.append(t)
      value = []
      for c in c_listL[cLi]:
        value.append(1.0*c_dic_l[cLi][(c,t)])
      valueL1.append(deepcopy(value))

      tem = stderror(value)

      #eelist.append(ee0[t]/(tem[1]**2))
      if inverse==0:
        if flagsn == 0:
          eelist.append(((tem[1])**2)/ee0[t])
        if flagsn == 1:
          eelist.append(((tem[1]/tem[0])**2)/ee0[t])
      if inverse==1:
        if flagsn == 0:
          eelist.append(ee0[t]/((tem[1])**2))
        if flagsn == 1:
          eelist.append(ee0[t]/((tem[1]/tem[0])**2))

    if simple == 1:
      pi_plot.a_line(xlist,eelist,lin="",label_l=r"$E^2$ %s"%(nameL[cLi]))

    if simple != 1:
      xlist = numpy.asarray(xlist) + cLi*0.02

      ylist = []
      elist = []
      #for t in t_list:
      for t in range(len(t_listL[cLi])):
      #for t in range(len(t_listL[0])):
        if rangex != []:
          if t >= rangex[1]:
            continue
        v0 = numpy.asarray(valueL0[t])
        v1 = numpy.asarray(valueL1[t])
        re = []
        re0 = stderror(v0)
        re1 = stderror(v1)
        re.append((re0[1]**2)/(re1[1]**2))
        for c in c_list:
          vc0 = numpy.delete(v0,c)
          vc1 = numpy.delete(v1,c)
          re0 = stderror(vc0)
          re1 = stderror(vc1)
          re.append((re0[1]**2)/(re1[1]**2))
        v = errorJack(re)
        ylist.append(v[0])
        elist.append(v[1])

      pi_plot.err_line(xlist,ylist,elist,lin="",label_l=r"$E^2$ %s"%(nameL[cLi]))

  pi_plot.plot_f()


