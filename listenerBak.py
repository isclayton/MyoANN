import socket
import array
import numpy as np
from matplotlib import pyplot as plt
import csv
import readchar
import select
import scipy as sp
from scipy import signal
import sounddevice as sdev
import multiprocessing as mp
#from matplotlib.mlab import PCA
import multiprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import audiogen
import pyaudio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#import pynput
import padasip as pa
import time
import pywt
class DATA:
    high = 20/(35/2)
    low = 450/(35/2)
    tone = audiogen.tone(480)
    training_time_offset = 100
    sampleCount = 10
    b, a = sp.signal.butter(6, [high,low], btype='bandpass')
    R1 = np.empty([7,])
    R2 = np.empty([7,])
    R3 = np.empty([7,])
    R4 = np.empty([7,])
    R5 = np.empty([7,])
    R6 = np.empty([7,])
    R7 = np.empty([7,])

    trainingData = np.empty([5, ])
    d = np.array([2,3,4,5,0])

    def getBuffer(self):
        return self.databuffer1, self.databuffer2, self.databuffer3, self.databuffer4, self.databuffer5, self.databuffer6, self.databuffer7, self.databuffer8

    def fft(self, data):
        return np.fft.fft(data).flatten()#np.array([np.fft.fft(data), self.mav(data), self.rms(data), self.sd(data)]).flatten()

    def measure_d(self, x):

        d = 2*x[0] + 1*x[1] - 1.5*x[2]
        return d

    def preprocess(self, data):
        return sp.signal.filtfilt(self.b, self.a, self.X-np.mean(data[-10:-1], axis=0), padlen=0)

    def pca(self, data):
        pca = PCA(n_components=1)
        return pca.fit(data).transform(data).flatten()

    def lda(self, data):
        lda = LinearDiscriminantAnalysis(n_components=8)
        return lda.fit(data, y=None).transform(data)

    def mav(self, data):
        return np.mean(np.abs(data[-10:]))

    def rms(self, x):
        return np.sqrt(np.vdot(x[-10:], x[-10:])/x[-10:].size)

    def variance(self, data):
        return np.var(data[-10:])

    def sd(self, data):
        return np.std(data[-10:])

    def zc(self, data):
        return ((data[:-1] * data[1:]) < 0).sum()

    def ssc(self, data):
        slope = np.diff(data[::5]) / np.diff(data[::5])
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
        sock2 = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
        sock3 = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

        sock.bind((UDP_IP, UDP_PORT1))
        sock2.bind((UDP_IP, UDP_PORT2))
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
                data2, addr2 = sock2.recvfrom(1024) # buffer size is 1024 bytes
                dataList = str(bytes(data)).strip("'").split('(')[1].split(')')[0].strip(' ').split(',')
                dataList2 = str(bytes(data2)).strip("'").split('(')[1].split(')')[0].strip(' ').split(',')
                MESSAGE = ((np.array([int(dataList[0].strip(" ")),int(dataList[1].strip(" ")),int(dataList[2].strip(" ")),
                                int(dataList[3].strip(" ")),int(dataList[4].strip(" ")),int(dataList[5].strip(" ")),
                                int(dataList[6].strip(" ")),int(dataList[7].strip(" ")),int(dataList2[0].strip(" ")),int(dataList2[1].strip(" ")),int(dataList2[2].strip(" ")),
                                int(dataList2[3].strip(" ")),int(dataList2[4].strip(" ")),int(dataList2[5].strip(" ")),
                                int(dataList2[6].strip(" ")),int(dataList2[7].strip(" "))])))

                Y = MESSAGE
                print Y,
                print Y.shape
                print self.X.shape
                Y = np.hstack((Y, self.rms(Y)))
                Y = np.hstack((Y, self.mav(Y)))
                Y = np.hstack((Y, self.sd(Y)))
                print Y.shape
                print self.X.shape
                self.X = np.vstack((self.X,Y))

               # dwt = pywt.dwt(self.preprocess(self.X), 'db2')
                if self.X.shape[0]>=5:

                    fft = self.X.flatten()
                    self.X = np.delete(self.X, 0, 0)
                    print fft.shape
                    if i < 1*self.training_time_offset:
                        print "Training Rest"
                        resting = fft#np.concatenate((resting, fft))#(resting + fft)/2
                        print resting.size
                        i += 1
                        print i
                    elif i < 2*self.training_time_offset:
                        print "Training Index"
                        index = fft#np.concatenate((index,fft))#/2
                        print index.size
                        i += 1
                        print i
                    elif i < 3*self.training_time_offset:
                        print "Training Middle"
                        middle = fft#np.concatenate((middle,fft))#/2
                        #print middle
                        i += 1
                        print i
                    elif i < 4*self.training_time_offset:
                        print "Training Ring"
                        ring = fft#np.concatenate((ring,fft))#/2
                        #print ring
                        i += 1
                        print i
                    elif i < 5*self.training_time_offset:
                        print "Training Pinky"
                        pinky = fft#np.concatenate((pinky,fft))
                        #print pinky
                        i += 1
                        print i

                    #elif i < 6*self.training_time_offset:
                    #    print "Training Thumb"
                    #    thumb = (thumb + fft)/2
                    #    print i
                    #    i += 1
                    elif i == 5*self.training_time_offset:
                        print index.flatten()[1:].size, middle.flatten()[1:].size, ring.flatten()[1:].size, pinky.flatten()[1:].size, resting.flatten()[1:].size
                        #print "TRAINING SET:\n ", type(thumb), type(index), type(middle), type(ring), type(pinky), type(resting)
                        self.trainingData = np.array([index.flatten(), middle.flatten(), ring.flatten(), pinky.flatten(), resting.flatten()])#, self.rest))(index.astype(list)),(ring.astype(list)),
                        print self.trainingData.size
                        print self.trainingData
                        scalar = StandardScaler()
                        print scalar.fit(self.trainingData.reshape(1,-1))
                        X_train = scalar.transform(self.trainingData.reshape(1,-1))
                        clf = MLPClassifier(solver='sgd', alpha=1e-5,activation='logistic',
                                     hidden_layer_sizes=(20, 20, 20), random_state=1, verbose=True)
                        i += 1
                        print clf.fit(X_train, self.d)
                        time.sleep(2)
                    elif i >= 5*self.training_time_offset+1:
                   # while True:
                        normFFT = scalar.transform(fft.reshape(1,-1))
                        y = clf.predict(normFFT)

                        x += 1
                        if x==1:
                            if y == 0:
                                print "                                         # REST (Unclassified)"

                            #elif y == 1:
                            #    print " # THUMB"
                            #    #audiogen.sampler.play(audiogen.beep(frequency=400, seconds=0.05))
                            #    #self.tone = audiogen.tone(480)

                            elif y == 2:
                                print "        # INDEX"
                                #self.tone = audiogen.tone(500)
                                #audiogen.sampler.play(audiogen.beep(frequency=440, seconds=0.05))
                            elif y == 3:
                                print "               # MIDDLE"
                                #self.tone = audiogen.tone(520)
                                #audiogen.sampler.play(audiogen.beep(frequency=480, seconds=0.05))

                            elif y == 4:
                                print "                        # RING"
                                #self.tone = audiogen.tone(540)
                                #audiogen.sampler.play(audiogen.beep(frequency=520, seconds=0.05))

                            elif y == 5:
                                print "                               # PINKY"
                                #self.tone = audiogen.tone(560)
                                #audiogen.sampler.play(audiogen.beep(frequency=560, seconds=0.05))

                            else: print "                                                # REST"
                            i += 1

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
                            x=0



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
