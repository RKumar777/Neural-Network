
# coding: utf-8

# In[28]:


import numpy as np
import sys
epoch=int(sys.argv[6])
ar1=np.genfromtxt(sys.argv[1],skip_header=0,delimiter=',',dtype=None,unpack=True)
ar2=np.genfromtxt(sys.argv[2],skip_header=0,delimiter=',',dtype=None,unpack=True)
nabla1=open(sys.argv[3],"w")
nabla2=open(sys.argv[4],"w")
nabla3=open(sys.argv[5],"w")
ar2=np.transpose(ar2)
ar1=np.transpose(ar1)
lbs2=ar2[:,0]
lbs=ar1[:,0]
learning_rate=float(sys.argv[9])
ar2=np.delete(ar2,0,axis=1)
ar1=np.delete(ar1,0,axis=1) # these are my inputs
features=len(ar1[0]) # number of features
# print(features)
lb_unq=np.unique(lbs)
labels=len(lb_unq)
examples=len(ar1)
exams2=len(ar2)
# print(examples)
hidden_units=int(sys.argv[7])
flag=int(sys.argv[8])
alpha=np.zeros((hidden_units,features+1))
beta=np.zeros((labels,hidden_units+1))
if(flag==1):
    alpha=np.random.uniform(-0.1,0.1,size=alpha.shape)
    beta=np.random.uniform(-0.1,0.1,size=beta.shape)
    
if(flag==2):
    alpha=np.zeros((hidden_units,features+1))
    beta=np.zeros((labels,hidden_units+1))
def linearforward(a,alp):
#     print("depep")
#     print(a)
    a=np.matrix(a)
    alp=np.matrix(alp)
    a=np.transpose(a)
    a=np.insert(a,0,1)
    b=alp*np.transpose(a)
#     print("saw")
#     print(b)
    return b

def linearforward2(a,alp):
#     print("depep2")
#     print(a)
    a=np.matrix(a)
    alp=np.matrix(alp)
    a=np.transpose(a)
    a=np.insert(a,0,1)
#     print("gund2")
#     print(a)
    b=alp*np.transpose(a)
#     print("saw2")
#     print(b)
    return b

def linearbackward(a,bet,gb):
#     print("linear")
#     print(a)
#     print(bet)
#     print(gb)
    a=np.transpose(a)
    a=np.insert(a,0,1)
    a=np.matrix(a)
    gb=np.matrix(gb)
    gb=np.transpose(gb)
#     print("jaggr")
#     print(a)
    gw=gb*a
    bett=np.transpose(bet)
    bett=np.matrix(bett)
#     print("golmaal")
#     print(np.shape(bett))
#     print(np.shape(gb))
    ga=bett*gb
    
    return gw,ga

def linearbackward2(a,bet,gb):
#     print("linear")
#     print(a)
#     print(bet)
#     print(gb)
    a=np.transpose(a)
    a=np.matrix(a)
    gb=np.matrix(gb)
    gb=np.transpose(gb)
    gb=np.delete(gb,0,axis=0)
#     print("jaggr")
#     print(a)
    a=np.insert(a,0,1)
    gw=gb*a
    bett=np.transpose(bet)
    bett=np.matrix(bett)
#     print("golmaal")
#     print(np.shape(bett))
#     print(np.shape(gb))
    ga=bett*gb
    
    return gw,ga

def sigmoidforward(a):
    b=1/(1+np.exp(-a))
    return b

def sigmoidbackward(b,gb):
#     print("siggu")
#     print(b)
    b=np.transpose(b)
    b=np.insert(b,0,1)
#     print(b)
    b=np.matrix(b)
    gb=np.transpose(gb)
    gb=np.matrix(gb)
#     print("deepa")
#     print(gb)
    dun=np.multiply(b,(1-b))
    ga=np.multiply(gb,dun)
#     print("khas")
#     print(np.shape(ga))
#     print(ga)
    return ga

def softmaxforward(a):
    b=np.exp(a)/np.sum(np.exp(a))
    return b

def softmaxbackward(a,b,gb):
    
    c=np.transpose(b)
    c=np.matrix(c)
    b=np.matrix(b)
    d=b*c
    e=np.transpose(gb)
    e=np.matrix(e)
#     print("jag")
# #     print(d)
#     print(np.diag(b))
    f=np.diag(b.A1)-d
#     print("tan tan")
#     print(f)
    f=np.matrix(f)
    ga=e*f
    return ga

def crossentropyforward(a,ahat):
    m=np.zeros(labels)
    m[a]=1
    m=np.transpose(m)
    m=np.matrix(m)
    b=(-m*(np.log(ahat)))
    return b

def crossentropybackward(a,ahat,b,gb):
    m=np.zeros(labels)
    m[a]=1
    
    m=np.matrix(m)
    m=np.transpose(m)
#     print("m")
#     print(m)
#     print("ahat")
#     print(ahat)
    mt=m/ahat
#     print("tom")
#     print(mt)
    ga=np.multiply(-gb,mt)
    return ga
    

def nnforward(x,y,alpine,betane):
    a=linearforward(x,alpine)
    z=sigmoidforward(a)
    b=linearforward(z,betane)
    yhat=softmaxforward(b)
    j=crossentropyforward(y,yhat)
    return [x,a,z,b,yhat,j]

def nnbackward(x,y,alpino,bitano):
    obt=nnforward(x,y,alpino,bitano)
    e,a,z,b,yhat,j=obt
    gj=1
    gyhat=crossentropybackward(y,yhat,j,gj)
    gb=softmaxbackward(b,yhat,gyhat)
    gbeta,gz=linearbackward(z,bitano,gb)
#     print("gbeta")
#     print(gbeta)
#     print("gz")
#     print(gz)
    ga=sigmoidbackward(z,gz)
    galpha,gx=linearbackward2(x,a,ga)
    return galpha,gbeta,j

for epo in range(epoch):
    hum=0
    for i in range(examples):
        x=ar1[i,:]
        y=lbs[i]
        galf,gbbie,jonty=nnbackward(x,y,alpha,beta)
        alpha=alpha-(learning_rate*galf)
        beta=beta-(learning_rate*gbbie)
    #for train
    for tipsy in range(examples):
        raw=ar1[tipsy,:]
        smack=lbs[tipsy]
        galf,gbbie,tum=nnbackward(raw,smack,alpha,beta)
        hum=hum+tum
    print(hum/examples)
    nabla3.write("epoch={} crossentropy(train): {}".format(epo+1,str(float(hum)/examples)))
    nabla3.write("\n")
    hum=0
    #for validation
    for jam in range(exams2):
        raw=ar2[jam,:]
        smack=lbs2[jam]
        q1,q2,q3,q4,q5,q6=nnforward(raw,smack,alpha,beta)
        hum=hum+q6
    print(hum/exams2)
    nabla3.write("epoch={} crossentropy(validation): {}".format(epo+1,str(float(hum)/exams2)))
    nabla3.write("\n")
err1=0.0
err2=0.0
for jp in range(examples):
    raw=ar1[jp,:]
    smack=lbs[jp]
    q1,q2,q3,q4,q5,q6=nnforward(raw,smack,alpha,beta)
    pre_lab_tr=np.argmax(q5)
    nabla1.write(str(pre_lab_tr))
    nabla1.write("\n")
    if(np.argmax(q5)!=smack):
        err1=err1+1
for hj in range(exams2):
    raw=ar2[hj,:]
    smack=lbs2[hj]
    q1,q2,q3,q4,q5,q6=nnforward(raw,smack,alpha,beta)
    pre_lab_val=np.argmax(q5)
    nabla2.write(str(pre_lab_val))
    nabla2.write("\n")
    if(np.argmax(q5)!=smack):
        err2=err2+1
err_train=err1/examples
err_val=err2/exams2
print(err_train)
print(err_val)
nabla3.write("error(train): {}".format(err_train))
nabla3.write("\n")
nabla3.write("error(validation): {}".format(err_val))
nabla1.close()
nabla2.close()
nabla3.close()

