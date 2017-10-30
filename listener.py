import socket
import array
import numpy as np
from matplotlib import pyplot as plt
import csv
import biosppy
import readchar
import select
import scipy as sp
from scipy import signal
import sounddevice as sdev
import multiprocessing as mp
#from matplotlib.mlab import PCA
import multiprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
import audiogen
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
import pyaudio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#import pynput
import padasip as pa
import time
import pywt
class DATA:
    rawBuffer = 4
    stackingBuffer = 1
    gestureCount = 5
    intensity = 0
    high = 20/(35/2)
    low = 450/(35/2)
    tone = audiogen.tone(480)
    training_time_offset = 100
    sampleCount = 10
    finalOutput = list()
    b, a = sp.signal.butter(6, [high,low], btype='bandpass')
    X = np.empty([5,])
    R1 = np.empty([(rawBuffer-1),])
    R2 = np.empty([(rawBuffer-1),])
    R3 = np.empty([(rawBuffer-1),])
    R4 = np.empty([(rawBuffer-1),])
    R5 = np.empty([(rawBuffer-1),])
    R6 = np.empty([(rawBuffer-1),])
    R7 = np.empty([(rawBuffer-1),])
    R8 = np.empty([(rawBuffer-1),])
    R9 = np.empty([(rawBuffer-1),])
    R10 = np.empty([(rawBuffer-1),])
    R11 = np.empty([(rawBuffer-1),])
    R12 = np.empty([(rawBuffer-1),])
    R13 = np.empty([(rawBuffer-1),])
    R14 = np.empty([(rawBuffer-1),])
    R15 = np.empty([(rawBuffer-1),])
    R16 = np.empty([(rawBuffer-1),])
    Y1 = np.empty([1,])
    Y2 = np.empty([1,])
    Y3 = np.empty([1,])
    Y4 = np.empty([1,])
    Y5 = np.empty([1,])
    Y6 = np.empty([1,])
    Y7 = np.empty([1,])
    Y8 = np.empty([1,])
    Y9 = np.empty([1,])
    I = np.empty([8,])
    r1 = np.empty([1,])
    r2 = np.empty([1,])
    r3 = np.empty([1,])
    r4 = np.empty([1,])
    r5 = np.empty([1,])
    r6 = np.empty([1,])
    r7 = np.empty([1,])
    r8 = np.empty([1,])
    r9 = np.empty([1,])
    r10 = np.empty([1,])
    r11 = np.empty([1,])
    r12 = np.empty([1,])
    r13 = np.empty([1,])
    r14 = np.empty([1,])
    r15 = np.empty([1,])
    r16 = np.empty([1,])
    scalar1 = StandardScaler()
    scalar2 = StandardScaler()
    scalar3 = StandardScaler()
    scalar4 = StandardScaler()
    scalar5 = StandardScaler()
    scalar6 = StandardScaler()
    scalar7 = StandardScaler()
    scalar8 = StandardScaler()
    scalar9 = StandardScaler()
    scalar10 = StandardScaler()
    scalar11 = StandardScaler()
    scalar12 = StandardScaler()
    scalar13 = StandardScaler()
    scalar14 = StandardScaler()
    scalar15 = StandardScaler()
    scalar16 = StandardScaler()
    scalar17 = StandardScaler()
    scalar18 = StandardScaler()
    trainingData1 = np.empty([gestureCount, ])
    trainingData2 = np.empty([gestureCount, ])
    trainingData3 = np.empty([gestureCount, ])
    trainingData4 = np.empty([gestureCount, ])
    trainingData5 = np.empty([gestureCount, ])
    trainingData6 = np.empty([gestureCount, ])
    trainingData7 = np.empty([gestureCount, ])
    trainingData8 = np.empty([gestureCount, ])
    trainingData9 = np.empty([gestureCount, ])
    trainingData10 = np.empty([gestureCount, ])
    trainingData11 = np.empty([gestureCount, ])
    trainingData12 = np.empty([gestureCount, ])
    trainingData13 = np.empty([gestureCount, ])
    trainingData14 = np.empty([gestureCount, ])
    trainingData15 = np.empty([gestureCount, ])
    trainingData16 = np.empty([gestureCount, ])
    trainingData17 = np.empty([gestureCount, ])

    d = np.array([1,2,3,4,5])#,6,7,8,9,10])

    def getBuffer(self):
        return self.databuffer1, self.databuffer2, self.databuffer3, self.databuffer4, self.databuffer5, self.databuffer6, self.databuffer7, self.databuffer8

    def fft(self, data):
        return np.fft.fft(data).flatten()#np.array([np.fft.fft(data), self.mav(data), self.rms(data), self.sd(data)]).flatten()

    def measure_d(self, x):

        d = 2*x[0] + 1*x[1] - 1.5*x[2]
        return d

    def preprocess(self, data):
        return sp.signal.filtfilt(self.b, self.a, data, padlen=0)

    def kurtosis(self, data):
        return sp.stats.kurtosis(data)
    def pca(self, data):
        pca = Incre(n_components=1)
        return abs(pca.fit(data).transform(data)[0])

    def lda(self, data):
        lda = LinearDiscriminantAnalysis(n_components=8)
        return lda.fit(data, y=None).transform(data)

    def mav(self, data):
        #print "MAV:", np.mean(np.abs(data[-10:]))
        return np.mean(np.abs(data))

    def rms(self, x):
       # print "RMS: ", np.sqrt(np.vdot(x[-10:], x[-10:])/x[-10:].size)
        return np.sqrt(np.vdot(x, x)/x.size)

    def variance(self, data):
       # print "VARIANCE:", np.var(data[-10:])
        return np.var(data)

    def sd(self, data):
       # print "STDEV:", np.std(data[-10:])
        return np.std(data)

    def zc(self, data):
       # print "ZC:",((data[:-1] * data[1:]) < 0).sum()
        return ((data[:-1] * data[1:]) < 0).sum()

    def ssc(self, data):
        slope = np.diff(data[::5]) / np.diff(data[::5])
        #print "Slope:",slope
        return slope

    def tonecheck(self):
        audiogen.sampler.play(self.tone)

    def main(self):

        plt.axis([0, 700, -200, 200])
        plt.ion()
        UDP_IP = "127.0.0.1"
        UDP_PORT1 = 8001
        UDP_PORT2 = 8000

        sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
        #sock2 = socket.socket(socket.AF_INET, # Internet
        #            socket.SOCK_DGRAM) # UDP
        #sock3 = socket.socket(socket.AF_INET, # Internet
        #             socket.SOCK_DGRAM) # UDP

        sock.bind((UDP_IP, UDP_PORT1))
       # sock2.bind((UDP_IP, UDP_PORT2))
        x = 0
        thumb =  np.empty([9595,])
        index = np.empty([9595,])
        middle = np.empty([9595,])
        ring = np.empty([9595,])
        pinky = np.empty([9595,])
        resting  = np.empty([9595,])
        plt.draw()
        i = 0
        with open('MyoData.csv', 'wb') as csvfile:

            while True:#i < 5*self.training_time_offset:

                data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes

                #data2, addr2 = sock2.recvfrom(1024) # buffer size is 1024 bytes
                if "/imu" not in str(data):
                    dataList = str(bytes(data)).strip("'").split('(')[1].split(')')[0].strip(' ').split(',')
                else:
                   # print data
                    continue
                #dataList2 = str(bytes(data2)).strip("'").split('(')[1].split(')')[0].strip(' ').split(',')
                MESSAGE = np.array([int(dataList[0].strip(" ")),int(dataList[1].strip(" ")),int(dataList[2].strip(" ")),
                                int(dataList[3].strip(" ")),int(dataList[4].strip(" ")),int(dataList[5].strip(" ")),
                                int(dataList[6].strip(" ")),int(dataList[7].strip(" "))])#,
                                #int(dataList2[0].strip(" ")),int(dataList2[1].strip(" ")),int(dataList2[2].strip(" ")),
                                #int(dataList2[3].strip(" ")),int(dataList2[4].strip(" ")),int(dataList2[5].strip(" ")),
                                #int(dataList2[6].strip(" ")),int(dataList2[7].strip(" "))])
                                #Two band data
                MESSAGE2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                self.intensity = self.variance(MESSAGE)
                #print MESSAGE
                #r1 = MESSAGE[0]
                #r2 = MESSAGE[1]
                #r3 = message[2]
                #r4 = MESSAGE[3]
                #r5 = MESSAGE[4]
                #r6 = MESSAGE[5]
                #r7 = MESSAGE[6]

                #print Y,
                #print Y.shape
                #print self.X .shape
                self.r1 = np.hstack((self.r1, MESSAGE[0]))
                self.r2 = np.hstack((self.r2, MESSAGE[1]))
                self.r3 = np.hstack((self.r3, MESSAGE[2]))
                self.r4 = np.hstack((self.r4, MESSAGE[3]))
                self.r5 = np.hstack((self.r5, MESSAGE[4]))
                self.r6 = np.hstack((self.r6, MESSAGE[5]))
                self.r7 = np.hstack((self.r7, MESSAGE[6]))
                self.r8 = np.hstack((self.r8, MESSAGE[7]))
                self.r9 = np.hstack((self.r9, MESSAGE2[8]))
                self.r10 = np.hstack((self.r10, MESSAGE2[9]))
                self.r11 = np.hstack((self.r11, MESSAGE2[10]))
                self.r12 = np.hstack((self.r12, MESSAGE2[11]))
                self.r13 = np.hstack((self.r13, MESSAGE2[12]))
                self.r14 = np.hstack((self.r14, MESSAGE2[13]))
                self.r15 = np.hstack((self.r15, MESSAGE2[14]))
                self.r16 = np.hstack((self.r16, MESSAGE2[15]))
                #print self.r1
                ##if self.r1.shape[0] == 50:
                #    self.r1 = np.delete(self.r1, 0, 1)
                #    self.r2 = np.delete(self.r2, 0, 1)
                #    self.r3 = np.delete(self.r3, 0, 1)
                #    self.r4 = np.delete(self.r4, 0, 1)
                #    self.r5 = np.delete(self.r5, 0, 1)
                #    self.r6 = np.delete(self.r6, 0, 1)
                #    self.r7 = np.delete(self.r7, 0, 1)
                #    self.r8 = np.delete(self.r8, 0, 1)
                #    self.r9 = np.delete(self.r9, 0, 1)
                ###    self.r10 = np.delete(self.r10, 0, 1)
                  #  self.r11 = np.delete(self.r11, 0, 1)
                  #  self.r12 = np.delete(self.r12, 0, 1)
                  #  self.r13 = np.delete(self.r13, 0, 1)
                  #  self.r14 = np.delete(self.r14, 0, 1)
                  #  self.r15 = np.delete(self.r15, 0, 1)
                  #  self.r16  = np.delete(self.r16, 0, 1)
                  #  print "r1 Array:", self.r1
                if len(self.r1) >= self.rawBuffer:
                    #print len(self.r1)
                    #emg = biosppy.emg.emg(biosppy.tools.filter_signal(self.r1.flatten(),order=6, frequency=4, sampling_rate=75)[0])
                  #  plt.draw()
                    #print emg
                    #self.plot(emg[0])
                   # print self.R1.shape
                    #temp1 =  np.array([self.mav(self.r1),self.rms(self.r1), self.variance(self.r1),self.sd(self.r1),self.zc(self.r1), self.kurtosis(self.r1)],dtype=np.float64)
                    #temp2 =  np.array([self.mav(self.r2),self.rms(self.r2), self.variance(self.r2),self.sd(self.r2),self.zc(self.r2), self.kurtosis(self.r2)],dtype=np.float64)
                    #temp3 =  np.array([self.mav(self.r3),self.rms(self.r3), self.variance(self.r3),self.sd(self.r3),self.zc(self.r3), self.kurtosis(self.r3)],dtype=np.float64)
                    #temp4 =  np.array([self.mav(self.r4),self.rms(self.r4), self.variance(self.r4),self.sd(self.r4),self.zc(self.r4), self.kurtosis(self.r4)],dtype=np.float64)
                    #temp5 =  np.array([self.mav(self.r5),self.rms(self.r5), self.variance(self.r5),self.sd(self.r5),self.zc(self.r5), self.kurtosis(self.r5)],dtype=np.float64)
                    ##temp6 =  np.array([self.mav(self.r6),self.rms(self.r6), self.variance(self.r6),self.sd(self.r6),self.zc(self.r6), self.kurtosis(self.r6)],dtype=np.float64)
                    #temp7 =  np.array([self.mav(self.r7),self.rms(self.r7), self.variance(self.r7),self.sd(self.r7),self.zc(self.r7), self.kurtosis(self.r7)],dtype=np.float64)
                    #temp8 =  np.array([self.mav(self.r8),self.rms(self.r8), self.variance(self.r8),self.sd(self.r8),self.zc(self.r8), self.kurtosis(self.r8)],dtype=np.float64)
                    #temp9 =  np.array([self.mav(self.r9),self.rms(self.r9), self.variance(self.r9),self.sd(self.r9),self.zc(self.r9), self.kurtosis(self.r9)],dtype=np.float64)
                    #temp10 =  np.array([self.mav(self.r10),self.rms(self.r10), self.variance(self.r10),self.sd(self.r10),self.zc(self.r10), self.kurtosis(self.r10)],dtype=np.float64)
                    #temp11 =  np.array([self.mav(self.r11),self.rms(self.r11), self.variance(self.r11),self.sd(self.r11),self.zc(self.r11), self.kurtosis(self.r11)],dtype=np.float64)
                    #temp12 =  np.array([self.mav(self.r12),self.rms(self.r12), self.variance(self.r12),self.sd(self.r12),self.zc(self.r12), self.kurtosis(self.r12)],dtype=np.float64)
                    #temp13 =  np.array([self.mav(self.r13),self.rms(self.r13), self.variance(self.r13),self.sd(self.r13),self.zc(self.r13), self.kurtosis(self.r13)],dtype=np.float64)
                    #temp14 =  np.array([self.mav(self.r14),self.rms(self.r14), self.variance(self.r14),self.sd(self.r14),self.zc(self.r14), self.kurtosis(self.r14)],dtype=np.float64)
                    #temp15 =  np.array([self.mav(self.r15),self.rms(self.r15), self.variance(self.r15),self.sd(self.r15),self.zc(self.r15), self.kurtosis(self.r15)],dtype=np.float64)
                    #temp16 =  np.array([self.mav(self.r16),self.rms(self.r16), self.variance(self.r16),self.sd(self.r16),self.zc(self.r16), self.kurtosis(self.r16)],dtype=np.float64)
                    #temp1 =  (np.array([(np.hstack((self.r1.flatten()[1:],np.array([self.variance(self.r1)]))))],dtype=np.float64)).flatten()
                    #temp2 =  (np.array([(np.hstack((self.r2.flatten()[1:],np.array([self.variance(self.r2)]))))],dtype=np.float64)).flatten()
                    #temp3 =  (np.array([(np.hstack((self.r3.flatten()[1:],np.array([self.variance(self.r3)]))))],dtype=np.float64)).flatten()
                    #temp4 =  (np.array([(np.hstack((self.r4.flatten()[1:],np.array([self.variance(self.r4)]))))],dtype=np.float64)).flatten()
                    #temp5 =  (np.array([(np.hstack((self.r5.flatten()[1:],np.array([self.variance(self.r5)]))))],dtype=np.float64)).flatten()
                    #temp6 =  (np.array([(np.hstack((self.r6.flatten()[1:],np.array([self.variance(self.r6)]))))],dtype=np.float64)).flatten()
                    #temp7 =  (np.array([(np.hstack((self.r7.flatten()[1:],np.array([self.variance(self.r7)]))))],dtype=np.float64)).flatten()
                    #temp8 =  (np.array([(np.hstack((self.r8.flatten()[1:],np.array([self.variance(self.r8)]))))],dtype=np.float64)).flatten()
                    #temp9 =  (np.array([(np.hstack((self.r9.flatten()[1:],np.array([self.variance(self.r9)]))))],dtype=np.float64)).flatten()
                    #temp10 =  (np.array([(np.hstack((self.r10.flatten()[1:],np.array([self.variance(self.r10)]))))],dtype=np.float64)).flatten()
                    #temp11 =  (np.array([(np.hstack((self.r11.flatten()[1:],np.array([self.variance(self.r11)]))))],dtype=np.float64)).flatten()
                    #temp12 =  (np.array([(np.hstack((self.r12.flatten()[1:],np.array([self.variance(self.r12)]))))],dtype=np.float64)).flatten()
                    #temp13 = ( np.array([(np.hstack((self.r13.flatten()[1:],np.array([self.variance(self.r13)]))))],dtype=np.float64)).flatten()
                    #temp14 =  (np.array([(np.hstack((self.r14.flatten()[1:],np.array([self.variance(self.r14)]))))],dtype=np.float64)).flatten()
                    #temp15 =  (np.array([(np.hstack((self.r15.flatten()[1:],np.array([self.variance(self.r15)]))))],dtype=np.float64)).flatten()
                    #temp16 =  (np.array([(np.hstack((self.r16.flatten()[1:],np.array([self.variance(self.r16)]))))],dtype=np.float64)).flatten()
                    temp1 = self.r1.flatten()[1:]#,np.array([self.variance(self.r1)]))))],dtype=np.float64)).flatten()
                    temp2 = self.r2.flatten()[1:]#,np.array([self.variance(self.r2)]))))],dtype=np.float64)).flatten()
                    temp3 = self.r3.flatten()[1:]#,np.array([self.variance(self.r3)]))))],dtype=np.float64)).flatten()
                    temp4 = self.r4.flatten()[1:]#,np.array([self.variance(self.r4)]))))],dtype=np.float64)).flatten()
                    temp5 = self.r5.flatten()[1:]#,np.array([self.variance(self.r5)]))))],dtype=np.float64)).flatten()
                    temp6 = self.r6.flatten()[1:]#,np.array([self.variance(self.r6)]))))],dtype=np.float64)).flatten()
                    temp7 = self.r7.flatten()[1:]#,np.array([self.variance(self.r7)]))))],dtype=np.float64)).flatten()
                    temp8 = self.r8.flatten()[1:]#,np.array([self.variance(self.r8)]))))],dtype=np.float64)).flatten()
                    temp9 = self.r9.flatten()[1:]#,np.array([self.variance(self.r9)]))))],dtype=np.float64)).flatten()
                    temp10 = self.r10.flatten()[1:]#,np.array([self.variance(self.r10)]))))],dtype=np.float64)).flatten()
                    temp11 = self.r11.flatten()[1:]#,np.array([self.variance(self.r11)]))))],dtype=np.float64)).flatten()
                    temp12 = self.r12.flatten()[1:]#,np.array([self.variance(self.r12)]))))],dtype=np.float64)).flatten()
                    temp13 = self.r13.flatten()[1:]#,np.array([self.variance(self.r13)]))))],dtype=np.float64)).flatten()
                    temp14 = self.r14.flatten()[1:]#,np.array([self.variance(self.r14)]))))],dtype=np.float64)).flatten()
                    temp15 = self.r15.flatten()[1:]#,np.array([self.variance(self.r15)]))))],dtype=np.float64)).flatten()
                    temp16 = self.r16.flatten()[1:]#,np.array([self.variance(self.r16)]))))],dtype=np.float64)).flatten()
                    del(self.r1)
                    del(self.r2)# = np.delete(self.r2, 0, 1)
                    del(self.r3)# = np.delete(self.r3, 0, 1)
                    del(self.r4)# = np.delete(self.r4, 0, 1)
                    del(self.r5)# = np.delete(self.r5, 0, 1)
                    del(self.r6)# = np.delete(self.r6, 0, 1)
                    del(self.r7)# = np.delete(self.r7, 0, 1)
                    del(self.r8)# = np.delete(self.r8, 0, 1)
                    del(self.r9)# = np.delete(self.r9, 0, 1)
                    del(self.r10)# = np.delete(self.r10, 0, 1)
                    del(self.r11)# = np.delete(self.r11, 0, 1)
                    del(self.r12)# = np.delete(self.r12, 0, 1)
                    del(self.r13)# = np.delete(self.r13, 0, 1)
                    del(self.r14)# = np.delete(self.r14, 0, 1)
                    del(self.r15)# = np.delete(self.r15, 0, 1)
                    del(self.r16)#  = np.delete(self.r16, 0, 1)
                    #print temp1.shape

                   # temp17 =  MESSAGE#np.array([MESSAGE[0][5]],dtype=np.float64)#self.mav(self.r6),self.rms(self.r6), self.variance(self.r6),self.sd(self.r6),self.zc(self.r6), self.kurtosis(self.r6)],dtype=np.float64)

                    self.R1 = np.vstack((self.R1, temp1))
                    self.R2 = np.vstack((self.R2, temp2))
                    self.R3 = np.vstack((self.R3, temp3))
                    self.R4 = np.vstack((self.R4, temp4))
                    self.R5 = np.vstack((self.R5, temp5))
                    self.R6 = np.vstack((self.R6, temp6))
                    self.R7 = np.vstack((self.R7, temp7))
                    self.R8 = np.vstack((self.R8, temp8))
                    self.R9 = np.vstack((self.R9, temp9))
                    self.R10 = np.vstack((self.R10, temp10))
                    self.R11 = np.vstack((self.R11, temp11))
                    self.R12 = np.vstack((self.R12, temp12))
                    self.R13 = np.vstack((self.R13, temp13))
                    self.R14 = np.vstack((self.R14, temp14))
                    self.R15 = np.vstack((self.R15, temp15))
                    self.R16 = np.vstack((self.R16, temp16))
                    #self.I = np.vstack((self.I, temp17))
                    #print "R1 Array Size:", self.R1
                    #if self.R1.size == 10:
                    #    self.R1 = np.delete(self.R1, 0, 0)
                    #self.R1 = np.hstack((self.R1, ))
                    #self.R1 = np.hstack((self.R1, ))
                    #self.R1 = np.hstack((self.R1, ))
                    #self.R1 = np.hstack((self.R1, ))
                    #self.R1 = np.hstack((self.R1, self.ssc(self.r1)))
                    #print "R1 Array:", self.R1
                    if self.R1.shape[0] >=self.stackingBuffer:#self.training_time_offset:
                        #print "R1 Array Full"
                        self.R1 = np.delete(self.R1, 0, 0)
                        self.R2 = np.delete(self.R2, 0, 0)
                        self.R3 = np.delete(self.R3, 0, 0)
                        self.R4 = np.delete(self.R4, 0, 0)
                        self.R5 = np.delete(self.R5, 0, 0)
                        self.R6 = np.delete(self.R6, 0, 0)
                        self.R7 = np.delete(self.R7, 0, 0)
                        self.R8 = np.delete(self.R8, 0, 0)
                        self.R9 = np.delete(self.R9, 0, 0)
                        self.R10 = np.delete(self.R10, 0, 0)
                        self.R11 = np.delete(self.R11, 0, 0)
                        self.R12 = np.delete(self.R12, 0, 0)
                        self.R13 = np.delete(self.R13, 0, 0)
                        self.R14 = np.delete(self.R14, 0, 0)
                        self.R15 = np.delete(self.R15, 0, 0)
                        self.R16 = np.delete(self.R16, 0, 0)
                        #self.I = np.delete(self.I, 0, 0)
                       # print "After:", self.R1

                        self.T1 = self.R1.flatten()
                        self.T2 = self.R2.flatten()
                        self.T3 = self.R3.flatten()
                        self.T4 = self.R4.flatten()
                        self.T5 = self.R5.flatten()
                        self.T6 = self.R6.flatten()
                        self.T7 = self.R7.flatten()
                        self.T8 = self.R8.flatten()
                        self.T9 = self.R9.flatten()
                        self.T10 = self.R10.flatten()
                        self.T11 = self.R11.flatten()
                        self.T12 = self.R12.flatten()
                        self.T13 = self.R13.flatten()
                        self.T14 = self.R14.flatten()
                        self.T15 = self.R15.flatten()
                        self.T16 = self.R16.flatten()
                        #print self.T1
                        #self.T17 = self.I.flatten()

                        if i<40:
                            del(self.R1 )#= np.delete(del(self.R1, 0, 0)
                            del(self.R2 )#= np.delete(del(self.R2, 0, 0)
                            del(self.R3 )#= np.delete(del(self.R3, 0, 0)
                            del(self.R4 )#= np.delete(del(self.R4, 0, 0)
                            del(self.R5 )#= np.delete(del(self.R5, 0, 0)
                            del(self.R6 )#= np.delete(del(self.R6, 0, 0)
                            del(self.R7 )#= np.delete(del(self.R7, 0, 0)
                            del(self.R8 )#= np.delete(del(self.R8, 0, 0)
                            del(self.R9 )#= np.delete(del(self.R9, 0, 0)
                            del(self.R10 )#= np.delete(del(self.R10, 0, 0)
                            del(self.R11 )#= np.delete(del(self.R11, 0, 0)
                            del(self.R12 )#= np.delete(del(self.R12, 0, 0)
                            del(self.R13 )#= np.delete(del(self.R13, 0, 0)
                            del(self.R14 )#= np.delete(del(self.R14, 0, 0)
                            del(self.R15 )#= np.delete(del(self.R15, 0, 0)
                            del(self.R16 )#= np.delete(del(self.R16, 0, 0)
                        #self.I = np.delete(self.I, 0, 0)
                        if i < 8:#1*self.training_time_offset:
                            print "Training Rest"

                            resting1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1 #np.concatenate((resting, fft))#(resting + fft)/2
                            print resting1
                            #resting2 = self.T2
                            #resting3 = self.T3
                            #resting4 = self.T4
                            #resting5 = self.T5
                            #resting6 = self.T6
                            #resting7 = self.T7
                            #resting8 = self.T8
                            #resting9 = self.T9
                            #resting10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #resting11 = self.T11
                            #resting12 = self.T12
                            #resting13 = self.T13
                            #resting14 = self.T14
                            #resting15 = self.T15
                            #resting16 = self.T16
                            #resting17 = self.T17
                            
                            #print resting.size
                            i += 1
                            print i

                        elif i < 16:#*self.training_time_offset:
                            print "Training Index"

                            index1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#np.concatenate((index,fft))#/2
                            #index2 = self.T2
                            #index3 = self.T3
                            #index4 = self.T4
                            #index5 = self.T5
                            #index6 = self.T6
                            #index7 = self.T7
                            #index8 = self.T8
                            #index9 = self.T9
                            #index10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #index11 = self.T11
                            #index12 = self.T12
                            #index13 = self.T13
                            #index14 = self.T14
                            #index15 = self.T15
                            #index16 = self.T16
                            #index17 = self.T17
                           # print index.size
                            i += 1
                            #print i
                        elif i < 24:#*self.training_time_offset:
                            print "Training Middle"

                            middle1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1#np.concatenate((middle,fft))#/2
                            #middle2 = self.T2
                            #middle3 = self.T3
                            #middle4 = self.T4
                            #middle5 = self.T5
                            #middle6 = self.T6
                            #middle7 = self.T7
                            #middle8 = self.T8
                            #middle9 = self.T9
                            #middle10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #middle11 = self.T11
                            #middle12 = self.T12
                            #middle13 = self.T13
                            #middle14 = self.T14
                            #middle15 = self.T15
                            #middle16 = self.T16
                           # middle17 = self.T17
                            #print middle
                            i += 1
                            #print i
                        elif i < 32:#*self.training_time_offset:
                            print "Training Ring"

                            
                            ring1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1#np.concatenate((ring,fft))#/2
                            #ring2 = self.T2
                            #ring3 = self.T3
                            #ring4 = self.T4
                            #ring5 = self.T5
                            #ring6 = self.T6
                            #ring7 = self.T7
                            #ring8 = self.T8
                            #ring9 = self.T9
                            #ring10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #ring11 = self.T11
                            #ring12 = self.T12
                            #ring13 = self.T13
                            #ring14 = self.T14
                            ##ring15 = self.T15
                            #ring16 = self.T16
                            ##ring17 = self.T17

                            #print ring
                            i += 1
                            #print i
                        elif i < 40:#*self.training_time_offset:
                            print "Training Pinky"

                            pinky1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1
                            #pinky2 = self.T2
                            #pinky3 = self.T3
                            #pinky4 = self.T4
                            #pinky5 = self.T5
                            #pinky6 = self.T6
                            #pinky7 = self.T7
                            #pinky8 = self.T8
                            #pinky9 = self.T9
                            #pinky10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #pinky11 = self.T11
                            #3pinky12 = self.T12
                            #pinky13 = self.T13
                            #pinky14 = self.T14
                            #pinky15 = self.T15
                            #pinky16 = self.T16
                           # pinky17 = self.T17
                            #np.concatenate((pinky,fft))
                            #print pinky
                            i += 1
                            #print i
                       # elif i < 24:#*self.training_time_offset:
                        #    print "Training Fist"

                         #   fist1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1
                            #pinky2 = self.T2
                            #pinky3 = self.T3
                            #pinky4 = self.T4
                            #pinky5 = self.T5
                            #pinky6 = self.T6
                            #pinky7 = self.T7
                            #pinky8 = self.T8
                            #pinky9 = self.T9
                            #pinky10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #pinky11 = self.T11
                            #3pinky12 = self.T12
                            #pinky13 = self.T13
                            #pinky14 = self.T14
                            #pinky15 = self.T15
                            #pinky16 = self.T16
                           # pinky17 = self.T17
                            #np.concatenate((pinky,fft))
                            #print pinky
                          #  i += 1
                            #print i
                        #elif i < 30:#*self.training_time_offset:
                        #    print "Training Peace"

                         #   peace1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1
                            #pinky2 = self.T2
                            #pinky3 = self.T3
                            #pinky4 = self.T4
                            #pinky5 = self.T5
                            #pinky6 = self.T6
                            #pinky7 = self.T7
                            #pinky8 = self.T8
                            #pinky9 = self.T9
                            #pinky10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #pinky11 = self.T11
                            #3pinky12 = self.T12
                            #pinky13 = self.T13
                            #pinky14 = self.T14
                            #pinky15 = self.T15
                            #pinky16 = self.T16
                           # pinky17 = self.T17
                            #np.concatenate((pinky,fft))
                            #print pinky
                         #   i += 1
                       ## elif i < 34:#*self.training_time_offset:
                         #   print "Training ThumbsUp"

                         #   thumbsup1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1
                            #pinky2 = self.T2
                            #pinky3 = self.T3
                            #pinky4 = self.T4
                            #pinky5 = self.T5
                            #pinky6 = self.T6
                            #pinky7 = self.T7
                            #pinky8 = self.T8
                            #pinky9 = self.T9
                            #pinky10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #pinky11 = self.T11
                            #3pinky12 = self.T12
                            #pinky13 = self.T13
                            #pinky14 = self.T14
                            #pinky15 = self.T15
                            #pinky16 = self.T16
                           # pinky17 = self.T17
                            #np.concatenate((pinky,fft))
                            #print pinky
                          #  i += 1
                            #print i   
                            #print i
                        #elif i < 6*self.training_time_offset:
                        #    print "Training Thumb"
                        #    thumb = (thumb + fft)/2
                        #    print i
                        #    i += 1
                        #elif i < 38:#*self.training_time_offset:
                        #    print "Training OK"

                         #   ok1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1
                            #pinky2 = self.T2
                            #pinky3 = self.T3
                            #pinky4 = self.T4
                            #pinky5 = self.T5
                            #pinky6 = self.T6
                            #pinky7 = self.T7
                            #pinky8 = self.T8
                            #pinky9 = self.T9
                            #pinky10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #pinky11 = self.T11
                            #3pinky12 = self.T12
                            #pinky13 = self.T13
                            #pinky14 = self.T14
                            #pinky15 = self.T15
                            #pinky16 = self.T16
                           # pinky17 = self.T17
                            #np.concatenate((pinky,fft))
                            #print pinky
                         #   i += 1
                            #print i
                       # elif i < 42:#*self.training_time_offset:
                        #    print "Training OPEN"

                          #  open1 = np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8])#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1
                            #pinky2 = self.T2
                            #pinky3 = self.T3
                            #pinky4 = self.T4
                            #pinky5 = self.T5
                            #pinky6 = self.T6
                            #pinky7 = self.T7
                            #pinky8 = self.T8
                            #pinky9 = self.T9
                            #pinky10 = self.T10 #np.concatenate((resting, fft))#(resting + fft)/2
                            #pinky11 = self.T11
                            #3pinky12 = self.T12
                            #pinky13 = self.T13
                            #pinky14 = self.T14
                            #pinky15 = self.T15
                            #pinky16 = self.T16
                           # pinky17 = self.T17
                            #np.concatenate((pinky,fft))
                            #print pinky
                          #  i += 1
                            #print i
                        elif i == 40:#:*self.training_time_offset:
                            #print index.flatten()[1:].size, middle.flatten()[1:].size, ring.flatten()[1:].size, pinky.flatten()[1:].size, resting.flatten()[1:].size
                           # print "TRAINING SET:\n ", type(thumb), type(index), type(middle), type(ring), type(pinky), type(resting)
                            self.trainingData1 = np.array([resting1.flatten(),index1.flatten(), middle1.flatten(), ring1.flatten(), pinky1.flatten()])#, fist1.flatten(), peace1.flatten(), thumbsup1.flatten(), ok1.flatten(), open1.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                           # self.trainingData2 = np.array([resting2.flatten(),index2.flatten(), middle2.flatten(), ring2.flatten(), pinky2.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData3 = np.array([resting3.flatten(),index3.flatten(), middle3.flatten(), ring3.flatten(), pinky3.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData4 = np.array([resting4.flatten(),index4.flatten(), middle4.flatten(), ring4.flatten(), pinky4.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData5 = np.array([resting5.flatten(),index5.flatten(), middle5.flatten(), ring5.flatten(), pinky5.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData6 = np.array([resting6.flatten(),index6.flatten(), middle6.flatten(), ring6.flatten(), pinky6.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData7 = np.array([resting7.flatten(),index7.flatten(), middle7.flatten(), ring7.flatten(), pinky7.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData8 = np.array([resting8.flatten(),index8.flatten(), middle8.flatten(), ring8.flatten(), pinky8.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData9 = np.array([resting9.flatten(),index9.flatten(), middle9.flatten(), ring9.flatten(), pinky9.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            ##self.trainingData10 = np.array([resting10.flatten(),index10.flatten(), middle10.flatten(), ring10.flatten(), pinky10.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData11 = np.array([resting11.flatten(),index11.flatten(), middle11.flatten(), ring11.flatten(), pinky11.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData12 = np.array([resting12.flatten(),index12.flatten(), middle12.flatten(), ring12.flatten(), pinky12.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData13 = np.array([resting13.flatten(),index13.flatten(), middle13.flatten(), ring13.flatten(), pinky13.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData14 = np.array([resting14.flatten(),index14.flatten(), middle14.flatten(), ring14.flatten(), pinky14.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            ##self.trainingData15 = np.array([resting15.flatten(),index15.flatten(), middle15.flatten(), ring15.flatten(), pinky15.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            #self.trainingData16 = np.array([resting16.flatten(),index16.flatten(), middle16.flatten(), ring16.flatten(), pinky16.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                           # self.trainingData17 = np.array([resting17.flatten(),index17.flatten(), middle17.flatten(), ring17.flatten(), pinky17.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            
                            #self.trainingData9 = np.array([[self.variance(resting9.flatten())],[self.variance(index9.flatten())], [self.variance(middle9.flatten())], [self.variance(ring9.flatten())], [self.variance(pinky9.flatten())]])#, self.rest))(index.astype(list)),(ring.astype(list)),
                            print "Training Data Shape:",self.trainingData1.shape#, self.trainingData1



                            self.scalar1.fit(self.trainingData1)
                            #self.scalar2.fit(self.trainingData2)
                            #self.scalar3.fit(self.trainingData3)
                            #self.scalar4.fit(self.trainingData4)
                            #self.scalar5.fit(self.trainingData5)
                            ##self.scalar6.fit(self.trainingData6)
                            #self.scalar7.fit(self.trainingData7)
                            #self.scalar8.fit(self.trainingData8)
                            ##self.scalar9.fit(self.trainingData9)
                            #self.scalar10.fit(self.trainingData10)
                            #self.scalar11.fit(self.trainingData11)
                            #self.scalar12.fit(self.trainingData12)
                            #self.scalar13.fit(self.trainingData13)
                            #self.scalar14.fit(self.trainingData14)
                            #self.scalar15.fit(self.trainingData15)
                            #self.scalar16.fit(self.trainingData16)
                           # self.scalar17.fit(self.trainingData17)


                            self.normT1 = self.scalar1.transform(self.trainingData1)
                            #self.normT2 = self.scalar2.transform(self.trainingData2)
                            #self.normT3 = self.scalar3.transform(self.trainingData3)
                            #self.normT4 = self.scalar4.transform(self.trainingData4)
                            #self.normT5 = self.scalar5.transform(self.trainingData5)
                            #self.normT6 = self.scalar6.transform(self.trainingData6)
                            #self.normT7 = self.scalar7.transform(self.trainingData7)
                            #self.normT8 = self.scalar8.transform(self.trainingData8)
                            #self.normT9 = self.scalar9.transform(self.trainingData9)

                            #self.normT10 = self.scalar10.transform(self.trainingData10)
                            #self.normT11 = self.scalar11.transform(self.trainingData11)
                            #self.normT12 = self.scalar12.transform(self.trainingData12)
                            #self.normT13 = self.scalar13.transform(self.trainingData13)
                            #self.normT14 = self.scalar14.transform(self.trainingData14)
                            ##self.normT15 = self.scalar15.transform(self.trainingData15)
                            #self.normT16 = self.scalar16.transform(self.trainingData16)
                           # self.normT17 = self.scalar17.transform(self.trainingData17)
                            

                           # print X_train
                           # clf1 = SGDClassifier(loss="modified_huber", penalty="l1")
                            #clf1 = NearestCentroid()
                            #clf1 = GaussianNB()
                            kernel = 1.0 * RBF(12*[2.0, 1.0])
                            clf1 = GaussianProcessClassifier(kernel=kernel)
                            #clf1 = tree.DecisionTreeClassifier()
                            #clf1 = OneVsOneClassifier(LinearSVC(random_state=0))
                            #clf1 = OneVsRestClassifier(LinearSVC(random_state=0))
                            #clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic',
                             #            hidden_layer_sizes=(5 ), shuffle=True, random_state=1, verbose=True)
                           # clf1 = OutputCodeClassifier(LinearSVC(random_state=0),
                                     # code_size=2, random_state=0)
                            #clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf3 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf4 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf6 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            ##clf7 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf8 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf9 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            ##             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf10 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf11 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf12 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            ##clf13 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                             #            hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf14 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf15 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf16 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                            #             hidden_layer_sizes=(10), random_state=1, verbose=True)
                            #clf17 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
                             #            hidden_layer_sizes=(2,2,2), random_state=1, verbose=True)
                            


                            clf1.fit(self.normT1, self.d)
                            #clf2.fit(self.normT2, self.d)
                            #clf3.fit(self.normT3, self.d)
                            #clf4.fit(self.normT4, self.d)
                            #clf5.fit(self.normT5, self.d)
                            #clf6.fit(self.normT6, self.d)
                            #clf7.fit(self.normT7, self.d)
                            #clf8.fit(self.normT8, self.d)
                            #clf9.fit(self.normT9, self.d)
                            #clf10.fit(self.normT10, self.d)
                            #clf11.fit(self.normT11, self.d)
                            #clf12.fit(self.normT12, self.d)
                            #clf13.fit(self.normT13, self.d)
                            #clf14.fit(self.normT14, self.d)
                            #clf15.fit(self.normT15, self.d)
                            #clf16.fit(self.normT16, self.d)
                            #clf17.fit(self.normT17, self.d)
                            i += 1

                        elif i >=40+1:# 5*self.training_time_offset+1:
                           # print "Before:"
                            #print "T1 Shape:", self.T1.shape
                            self.normT1 = self.scalar1.transform(np.array([self.T1,self.T2,self.T3,self.T4,self.T5,self.T6,self.T7,self.T8]).flatten().reshape(1,-1))#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1)

                            #self.normT2 = self.scalar2.transform(self.T2.reshape(1,-1))
                            #self.normT3 = self.scalar3.transform(self.T3.reshape(1,-1))
                            #self.normT4 = self.scalar4.transform(self.T4.reshape(1,-1))
                            #self.normT5 = self.scalar5.transform(self.T5.reshape(1,-1))
                            #self.normT6 = self.scalar6.transform(self.T6.reshape(1,-1))
                            #self.normT7 = self.scalar7.transform(self.T7.reshape(1,-1))
                            #self.normT8 = self.scalar8.transform(self.T8.reshape(1,-1))
                            #self.normT9 = self.scalar9.transform(self.T9.reshape(1,-1))
                            #self.normT10 = self.scalar10.transform(self.T10.reshape(1,-1))
                            #self.normT11 = self.scalar11.transform(self.T11.reshape(1,-1))
                            #self.normT12 = self.scalar12.transform(self.T12.reshape(1,-1))
                            #self.normT13 = self.scalar13.transform(self.T13.reshape(1,-1))
                            #self.normT14 = self.scalar14.transform(self.T14.reshape(1,-1))
                            #self.normT15 = self.scalar15.transform(self.T15.reshape(1,-1))
                            #self.normT16 = self.scalar16.transform(self.T16.reshape(1,-1))
                           ## self.normT17 = self.scalar17.transform(self.T17.reshape(1,-1))
                          # # print "After"
                       # whi#le True:
                            #normFFT = scalar.transform(fft.reshape(1,-1))
                            self.y1 = clf1.predict(self.preprocess(self.normT1))
                            #x1 = clf1.predict_proba(self.normT1)
                            #print x1
                            #self.y2 = clf2.predict(self.normT2)
                            #self.y3 = clf3.predict(self.normT3)
                            #self.y4 = clf4.predict(self.normT4)
                            #self.y5 = clf5.predict(self.normT5)
                            #self.y6 = clf6.predict(self.normT6)
                            #self.y7 = clf7.predict(self.normT7)
                            #self.y8 = clf8.predict(self.normT8)
                            #self.y9 = clf9.predict(self.normT9)
                            #self.y10 = clf10.predict(self.normT10)
                            #self.y11 = clf11.predict(self.normT11)
                            #self.y12 = clf12.predict(self.normT12)
                            #self.y13 = clf13.predict(self.normT13)
                            #self.y14 = clf14.predict(self.normT14)
                            #self.y15 = clf15.predict(self.normT15)
                            #self.y16 = clf16.predict(self.normT16)
                           ## self.y17 = clf17.predict(self.normT17)
                            
                            #print self.intensity
                            if self.intensity<1000:
                                intense = "*"
                            elif self.intensity<2000:
                                intense = "**"
                            elif self.intensity<3000:
                                intense = "***"
                            elif self.intensity<5000:
                                intense = "****"
                            elif self.intensity<9000:
                                intense = "****"
                            elif self.intensity<10000:
                                intense = "*****"
                            elif self.intensity<11000:
                                intense = "******"
                            elif self.intensity<13000:
                                intense = "*******"
                            elif self.intensity<15000:
                                intense = "********"
                            elif self.intensity<19000:
                                intense = "*********"
                            elif self.intensity<24000:
                                intense = "**********"
                            elif self.intensity<29000:
                                intense = "***********"
                            elif self.intensity<35000:
                                intense = "************"
                            elif self.intensity<40000:
                                intense = "*************"
                            elif self.intensity<45000:
                                intense = "**************"
                            elif self.intensity<50000:
                                intense = "***************"
                            elif self.intensity<55000:
                                intense = "****************"
                            elif self.intensity<60000:
                                intense = "******************"
                            elif self.intensity<65000:
                                intense = "*******************"
                            elif self.intensity<70000:
                                intense = "*********************"
                            elif self.intensity<80000:
                                intense = "***********************"
                            elif self.intensity<90000:
                                intense = "*************************"
                            elif self.intensity<100000:
                                intense = "***************************"
                            elif self.intensity<110000:
                                intense = "********************************"
                            elif self.intensity<120000:
                                intense = "*************************************"
                            elif self.intensity<130000:
                                intense = "******************************************"
                            else:
                                intense = "***************************************************"
                            #print self.y1#, self.y2,self.y3, self.y4, self.y5, self.y6, self.y7, self.y8, self.y9, self.y10, self.y11, self.y12, self.y13, self.y14, self.y15, self.y16
                            self.finalOutput.append(int(round((self.y1))))# + self.y2 + self.y3 + self.y4 + self.y5 + self.y6+ self.y7 + self.y8 +self.y9+ self.y10+ self.y11+ self.y12+ self.y13+ self.y14+ self.y15+ self.y16)/16))
                            if len(self.finalOutput) == 5:
                                #print self.finalOutput
                                #print "Length Matches"
                                print intense

                                if (self.finalOutput[0] + self.finalOutput[1] + self.finalOutput[2] + self.finalOutput[3] + self.finalOutput[4])/5 == self.finalOutput[0]:
                                  #  print "All the same"
                                   # print self.finalOutput
                                    if self.finalOutput[0] == 1:
                                        print "\n                                    "

                                    #elif self.finalOutput[0] == 1:
                                    #    print "\n # THUMB"
                                    #        #    #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=400, seconds=0.05))
                                    #        #    #self.tone = audiogen.tone(480)

                                    elif self.finalOutput[0] == 2:
                                        print "\n        # INDEX"
                                                #self.tone = audiogen.tone(500)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=440, seconds=0.05))
                                    elif self.finalOutput[0] == 3:
                                        print "\n               # MIDDLE"
                                                #self.tone = audiogen.tone(520)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=480, seconds=0.05))

                                    elif self.finalOutput[0] == 4:
                                        print "\n                        # RING"
                                                #self.tone = audiogen.tone(540)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=520, seconds=0.05))

                                    elif self.finalOutput[0] == 5:
                                        print " \n                              # PINKY"
                                                #self.tone = audiogen.tone(560)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencY=560, seconds=0.05))
                                    elif self.finalOutput[0] == 6:
                                        print #intense, "\n #                                            FIST"
                                            #    #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=400, seconds=0.05))
                                            #    #self.tone = audiogen.tone(480)

                                    elif self.finalOutput[0] == 7:
                                        print #intense, "\n                                                       # PEACE"
                                                #self.tone = audiogen.tone(500)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=440, seconds=0.05))
                                    elif self.finalOutput[0] == 8:
                                        print #intense, "\n                                                                 # THUMBS UP"
                                                #self.tone = audiogen.tone(520)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=480, seconds=0.05))

                                    elif self.finalOutput[0] == 9:
                                        print #intense, "\n                                                                                 #OK"
                                                #self.tone = audiogen.tone(540)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=520, seconds=0.05))

                                    elif self.finalOutput[0] == 10:
                                        print #intense, " \n                                                                                    # OPEN"
                                                #self.tone = audiogen.tone(560)
                                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencY=560, seconds=0.05))

                                    else: print intense, "\n                                                # REST"
                                            #i += 1
                                self.finalOutput.pop(0)
                                #
                            else:
                                pass
                                     #  "No NaNs"
                          #  self.Y1 = np.vstack((self.Y1, self.y1))
                          #  self.Y2 = np.vstack((self.Y2, self.y2))
                          #  self.Y3 = np.vstack((self.Y3, self.y3))
                          #  self.Y4 = np.vstack((self.Y4, self.y4))
                          #  self.Y5 = np.vstack((self.Y5, self.y5))
                          #  self.Y6 = np.vstack((self.Y6, self.y6))
                          #  self.Y7 = np.vstack((self.Y7, self.y7))
                          #  self.Y8 = np.vstack((self.Y8, self.y8))
                          #  self.Y9 = np.vstack((self.Y9, self.y9))
                            #if self.R1.size == 10:
                            #    self.R1 = np.delete(self.R1, 0, 0)
                            #self.R1 = np.hstack((self.R1, ))
                            #self.R1 = np.hstack((self.R1, ))
                            #self.R1 = np.hstack((self.R1, ))
                            #self.R1 = np.hstack((self.R1, ))
                            #self.R1 = np.hstack((self.R1, self.ssc(self.r1)))
                            #print "R1 Array:", self.R1
                          #  if self.Y1.shape[0] >=10:#self.training_time_offset:
                          #     # print "Before:", self.R1
                          #      self.Y1 = np.delete(self.Y1, 0, 0)
                          #      self.Y2 = np.delete(self.Y2, 0, 0)
                          #      self.Y3 = np.delete(self.Y3, 0, 0)
                          ##      self.Y4 = np.delete(self.Y4, 0, 0)
                           #     self.Y5 = np.delete(self.Y5, 0, 0)
                           #     self.Y6 = np.delete(self.Y6, 0, 0)
                           #     self.Y7 = np.delete(self.Y7, 0, 0)
                           #     self.Y8 = np.delete(self.Y8, 0, 0)
                           #     self.Y9 = np.delete(self.Y9, 0, 0)


                        #    i += 1
                        #    if i < 10*self.training_time_offset:
                        #        print "Training Final Rest"
                        #        resting1 = np.array([self.Y1.flatten(), self.Y2.flatten(), self.Y3.flatten(), self.Y4.flatten(), self.Y5.flatten(), self.Y6.flatten(), self.Y7.flatten(), self.Y8.flatten(),self.Y9.flatten()]) #np.concatenate((resting, fft))#(resting + fft)/2
                        #        print resting1
                        #        i += 1
                        #        #print i
                        #    elif i < 12*self.training_time_offset:
                        #        print "Training Final Index"
                        #        index1 = np.array([self.Y1.flatten(), self.Y2.flatten(), self.Y3.flatten(), self.Y4.flatten(), self.Y5.flatten(), self.Y6.flatten(), self.Y7.flatten(), self.Y8.flatten(),self.Y9.flatten()])
                        #        print index1
                        #        i += 1
                        #        #print i
                        #    elif i < 14*self.training_time_offset:
                        #        print "Training Final Middle"
                        #        middle1 = np.array([self.Y1.flatten(), self.Y2.flatten(), self.Y3.flatten(), self.Y4.flatten(), self.Y5.flatten(), self.Y6.flatten(), self.Y7.flatten(), self.Y8.flatten(),self.Y9.flatten()])
                        #        print middle1
                       #         i += 1
                                #print i
                      #      elif i < 16*self.training_time_offset:
                     #           print "Training Final Ring"
                    #            ring1 = np.array([self.Y1.flatten(), self.Y2.flatten(), self.Y3.flatten(), self.Y4.flatten(), self.Y5.flatten(), self.Y6.flatten(), self.Y7.flatten(), self.Y8.flatten(),self.Y9.flatten()])
                   #             print ring1
                  #              i += 1
                                #print i
                 #           elif i < 18*self.training_time_offset:
                #                print "Training Final Pinky"
               #                 pinky1 = np.array([self.Y1.flatten(), self.Y2.flatten(), self.Y3.flatten(), self.Y4.flatten(), self.Y5.flatten(), self.Y6.flatten(), self.Y7.flatten(), self.Y8.flatten(),self.Y9.flatten()])
              #                  print pinky1
                                #np.concatenate((pinky,fft))
             #                   print pinky
                               # i += 1


            #                elif i == 18*self.training_time_offset:

           #                     self.finalTrainingData1 = np.array([resting1.flatten(),index1.flatten(), middle1.flatten(), ring1.flatten(), pinky1.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
          #                      self.finalScalar1 = StandardScaler()
         #                       self.finalScalar1.fit(self.finalTrainingData1)
        #                        self.finalNormT1 = self.finalScalar1.transform(self.finalTrainingData1)
                            #    print self.finalTrainingData1

                               # print X_train
       #                         finalClf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,activation='logistic',
      #                                       hidden_layer_sizes=(10), random_state=1, verbose=True)

                                #finalD = np.array([1,2,3,4,5,6])
                                #finalClf1.n_outputs_=1
     #                           finalClf1.fit(self.finalTrainingData1, self.d)

    #                            i += 1

   #                         elif i >= 18*self.training_time_offset+1:
   #                             newData = np.array([self.y1[0], self.y2[0], self.y3[0], self.y4[0], self.y5[0], self.y6[0], self.y7[0], self.y8[0], self.y9[0]])
   #                             print newData
  #                              finalOutput = finalClf1.predict(newData.reshape(1,-1))
  #                              print finalOutput

                            #print y1, y2, y3, y4, y5, y6

 #                               x += 1
 #                               if x==1:

 #                                   if finalOutput == 1:
 #                                       print "                                         # REST (Unclassified)"

#                                    elif finalOutput == 1:
#                                        print " # THUMB"
                                    #    #audiogen.sampler.plafinalOutput(audiogen.beep(frequencfinalOutput=400, seconds=0.05))
                                    #    #self.tone = audiogen.tone(480)

#                                    elif finalOutput == 2:
#                                        print "        # INDEX"
                                        #self.tone = audiogen.tone(500)
                                        #audiogen.sampler.plafinalOutput(audiogen.beep(frequencfinalOutput=440, seconds=0.05))
#                                    elif finalOutput == 3:
#                                        print "               # MIDDLE"
                                        #self.tone = audiogen.tone(520)
                                        #audiogen.sampler.plafinalOutput(audiogen.beep(frequencfinalOutput=480, seconds=0.05))

#                                    elif finalOutput == 4:
#                                        print "                        # RING"
                                        #self.tone = audiogen.tone(540)
                                        #audiogen.sampler.plafinalOutput(audiogen.beep(frequencfinalOutput=520, seconds=0.05))
#
#                                    elif finalOutput == 5:
#                                        print "                               # PINKY"
#                                        #self.tone = audiogen.tone(560)
#                                        #audiogen.sampler.plafinalOutput(audiogen.beep(frequencY=560, seconds=0.05))
#
#                                    else: print "                                                # REST"
                              #      i += 1

                                  #  writer = csv.writer(csvfile, delimiter=',')
                                    #print self.X
                                   # fft = self.fft(self.X)
                                    #scalar = StandardScaler()
                                    #scalar.fit(fft)
                                    #normFFT = scalar.transform(fft)
                                   # self.plot(dwt)
                                    #print normFFT
                                    #print np.average(emg_filtered, 0)
                                    #writer.writerow(np.average(self.X, 0))
                                    #print np.sum(self.X, 0)/self.X.shape[0] - self.rest
                                    #self.plot( emg_filtered)#self.X)#self.X.shape[0])
                                   # x=0

                        else:
                            pass

                        #self.R1 = np.hstack((self.R1, self.rms(R1)))
                        #Y = np.hstack((Y, self.mav(Y)))
                        #Y = np.hstack((Y, self.sd(Y)))
                        #print Y.shape
                        #print self.X.shape
                        #self.X = np.vstack(self.R1)
                        #print "X Array:",self.X
                       # dwt = pywt.dwt(self.preprocess(self.X), 'db2')



    def plot(self, x):
            plt.draw()
            #pc = self.pca(self.)
           # print pc[-1][1]/pc[-1][0]
            plt.plot(x)
            #printft
            #fft = self.fft(x)
            #plt.plot(fft)
            #print fft
            #plt.plot(np.sum(self.X,1))
            #plt.plot((np.sum(self.X, 0)/self.X.shape[0]) - self.rest)
            #print#np.sum(self.X,0)/500
            #print np.linalg.norm(self.X, axis=None)#np.correlate(np.linalg.norm(self.X[-1,:], axis=None), np.linalg.norm(self.rest, axis=None)), np.correlate(np.linalg.norm( self.X[-1,:], axis=None),np.linalg.norm( self.index, axis=None))#, np.correlate(np.sum(self.X,0)/5, self.index) # np.correlate(self.X[-1,:], self.rest), np.correlate( np.sum(self.X[-1:-10,:])/10, self.index)
            #plt.axis([0, 1000, -200, 200])
            plt.show()
            plt.pause(0.00000000000000000000000001)
            plt.clf()
if __name__ == '__main__':

    DATA().main()
