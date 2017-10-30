import socket
import array
import numpy as np
from matplotlib import pyplot as plt
import csv
from sklearn import datasets
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
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
    initial = 50
    rawBuffer = 4
    stackingBuffer = 1
    gestureCount = 5
    intensity = 0
    high = 20/(35/2)
    low = 450/(35/2)
    tone = audiogen.tone(480)
    training_time_offset = 100
    sampleCount = 10
    finalOutput = [1]

    NUMBER_OF_BANDS = 2
    daNUMBER_OF_BANDS = 2
    indexLabel = 2
    middleLabel = 3
    ringLabel = 4
    pinkyLabel = 5
    restLabel = 1

    training = np.empty([8*NUMBER_OF_BANDS,])
    labels = np.empty([1,])
    resting = np.empty([8*NUMBER_OF_BANDS,])
    index = np.empty([8*NUMBER_OF_BANDS,])
    middle = np.empty([8*NUMBER_OF_BANDS,])
    ring = np.empty([8*NUMBER_OF_BANDS,])
    pinky = np.empty([8*NUMBER_OF_BANDS,])
    scalar1 = StandardScaler()

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


        if self.NUMBER_OF_BANDS == 2:
            sock2 = socket.socket(socket.AF_INET, # Internet
                    socket.SOCK_DGRAM) # UDP
            sock2.bind((UDP_IP, UDP_PORT2))

        sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
        sock.bind((UDP_IP, UDP_PORT1))

        x = 0

        i = 0
        with open('MyoData.csv', 'wb') as csvfile:

            while True:#i < 5*self.training_time_offset:

                data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
                dataList = str(bytes(data)).strip("'").split('(')[1].split(')')[0].strip(' ').split(',')
                if self.NUMBER_OF_BANDS == 2:
                    data2, addr2 = sock2.recvfrom(1024) # buffer size is 1024 bytes

                    dataList2 = str(bytes(data)).strip("'").split('(')[1].split(')')[0].strip(' ').split(',')
                #if "/imu" not in str(data):


               # else:
                   # print data
                   # continue
                #dataList2 = str(bytes(data2)).strip("'").split('(')[1].split(')')[0].strip(' ').split(',')
                if self.NUMBER_OF_BANDS == 2:
                    MESSAGE = np.array([int(dataList[0].strip(" ")),int(dataList[1].strip(" ")),int(dataList[2].strip(" ")),
                                int(dataList[3].strip(" ")),int(dataList[4].strip(" ")),int(dataList[5].strip(" ")),
                                int(dataList[6].strip(" ")),int(dataList[7].strip(" ")),
                                int(dataList2[0].strip(" ")),int(dataList2[1].strip(" ")),int(dataList2[2].strip(" ")),
                                int(dataList2[3].strip(" ")),int(dataList2[4].strip(" ")),int(dataList2[5].strip(" ")),
                                int(dataList2[6].strip(" ")),int(dataList2[7].strip(" "))])
                                #Two band data
                else:
                    MESSAGE = np.array([int(dataList[0].strip(" ")),int(dataList[1].strip(" ")),int(dataList[2].strip(" ")),
                                int(dataList[3].strip(" ")),int(dataList[4].strip(" ")),int(dataList[5].strip(" ")),
                                int(dataList[6].strip(" ")),int(dataList[7].strip(" "))])

              #  MESSAGE2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                self.intensity = self.variance(MESSAGE)
                if i < self.initial:
                    print "Get Ready for rest"
                    i += 1

                elif i < 5*self.initial:
                    print "Training Rest"
                    self.training = np.vstack((self.training, MESSAGE))
                    self.labels = np.hstack((self.labels, self.restLabel))
                    i += 1
                    print i

                elif i < 6*self.initial:
                    print "Get Ready for index"
                    i += 1

                elif i < 11*self.initial:
                    print "Training Index"

                    self.training = np.vstack((self.training, MESSAGE))
                    self.labels = np.hstack((self.labels, self.indexLabel))

                    i += 1
                    print i
                elif i < 12*self.initial:
                    print "Get Ready for middle"
                    i += 1

                elif i <17*self.initial:
                    print "Training Middle"
                    self.training = np.vstack((self.training, MESSAGE))
                    self.labels = np.hstack((self.labels, self.middleLabel))
                    i += 1
                    print i

                elif i < 18*self.initial:
                    print "Get Ready for ring"
                    i += 1

                elif i < 22*self.initial:
                    print "Training Ring"
                    self.training = np.vstack((self.training, MESSAGE))
                    self.labels = np.hstack((self.labels, self.ringLabel))
                    i += 1
                    print i
                elif i < 23*self.initial:
                    print "Get Ready for pinky"
                    i += 1

                elif i < 27*self.initial:
                    print "Training Pinky"
                    self.training = np.vstack((self.training, MESSAGE))
                    self.labels = np.hstack((self.labels, self.pinkyLabel))
                    i += 1
                    print i
                elif i ==27*self.initial:
                    self.training = self.training[1:]
                    self.labels = self.labels[1:]
                    print "Training: \n", self.training
                    print "Labels: \n", self.labels
                    self.scalar1.fit(self.training)
                    self.normT1 = self.scalar1.transform(self.training)

                    #clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic',
                      #          hidden_layer_sizes=(5,6), shuffle=True, random_state=1, verbose=True)
                    #kernel = 1.0 * RBF([8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0])
                    #clf1 = GaussianProcessClassifier(kernel=kernel)
                    #clf1 = tree.DecisionTreeClassifier()
                    clf1 = RandomForestClassifier(n_estimators=10)
                    clf1.fit(self.normT1, self.labels)

                    i += 1

                elif i >27*self.initial:# 5*self.training_time_offset+1:

                    self.normT1 = self.scalar1.transform(MESSAGE.reshape(1,-1))#,self.T9,self.T10,self.T11,self.T12,self.T13,self.T14,self.T15,self.T16])#self.T1)
                    self.y1 = clf1.predict(self.normT1)#self.scalar1.transform((3*MESSAGE)))#self.preprocess(self.normT1))
                    print self.y1

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
                    #print self.intensity
                    #self.finalOutput.append(int(round((self.y1))))# + self.y2 + self.y3 + self.y4 + self.y5 + self.y6+ self.y7 + self.y8 +self.y9+ self.y10+ self.y11+ self.y12+ self.y13+ self.y14+ self.y15+ self.y16)/16))
                   # if len(self.finalOutput) == 5:
                        #print self.finalOutput
                        #print "Length Matches"
                    print intense

                        #if (self.y1 + self.finalOutput[1] + self.finalOutput[2] + self.finalOutput[3] + self.finalOutput[4])/5 == self.y1:
                          #  print "All the same"
                           # print self.finalOutput
                    if self.y1 == 1:
                        print "\n                                    "

                    #elif self.y1 == 1:
                    #    print "\n # THUMB"
                    #        #    #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=400, seconds=0.05))
                    #        #    #self.tone = audiogen.tone(480)

                    elif self.y1 == 2:
                        print "\n        # INDEX"
                                #self.tone = audiogen.tone(500)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=440, seconds=0.05))
                    elif self.y1 == 3:
                        print "\n               # MIDDLE"
                                #self.tone = audiogen.tone(520)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=480, seconds=0.05))

                    elif self.y1 == 4:
                        print "\n                        # RING"
                                #self.tone = audiogen.tone(540)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=520, seconds=0.05))

                    elif self.y1 == 5:
                        print " \n                              # PINKY"
                                #self.tone = audiogen.tone(560)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencY=560, seconds=0.05))
                    elif self.y1 == 6:
                        print #intense, "\n #                                            FIST"
                            #    #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=400, seconds=0.05))
                            #    #self.tone = audiogen.tone(480)

                    elif self.y1 == 7:
                        print #intense, "\n                                                       # PEACE"
                                #self.tone = audiogen.tone(500)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=440, seconds=0.05))
                    elif self.y1 == 8:
                        print #intense, "\n                                                                 # THUMBS UP"
                                #self.tone = audiogen.tone(520)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=480, seconds=0.05))

                    elif self.y1 == 9:
                        print #intense, "\n                                                                                 #OK"
                                #self.tone = audiogen.tone(540)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencself.finalOutput=520, seconds=0.05))

                    elif self.y1 == 10:
                        print #intense, " \n                                                                                    # OPEN"
                                #self.tone = audiogen.tone(560)
                                #audiogen.sampler.plaself.finalOutput(audiogen.beep(frequencY=560, seconds=0.05))

                    else: print intense, "\n                                                # REST"
                            #i += 1
                #self.finalOutput.pop(0)
                        #
                    #else:
                    #    pass


if __name__ == '__main__':

    DATA().main()

