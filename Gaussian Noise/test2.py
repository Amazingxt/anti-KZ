# -*- coding: utf-8 -*-
# ==============================================================================
# FSR scaning for FP cavity in QNG
# Author: jmcui
# Date:   2017-4-1s
# Mail:   jmcui@mail.ustc.edu.cn
# ==============================================================================

import sys
from PyDAQmx import Task
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxTypes import *
from PyDAQmx.DAQmxCallBack import *
from PyDAQmx.DAQmxFunctions import *
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from pyqtgraph.ptime import time
import visa
import numpy as np


# Config DAQ Hardware


class DAQ_main:
    DAQ_drive = Task()
    DAQ_signal = Task()
    Triger = Task()
    TempFile = None

    # Amp: Voltage amplitude, default Triangle
    def __init__(self, SampleN=8000, Rate=500000.0):
        self.Rate = Rate
        self.SampsN = SampleN
        self.data = np.zeros((self.SampsN,), dtype=np.float64)

        self.DAQ_drive.CreateAOVoltageChan(
            "Dev2/ao0", "drive", 0, 10.0, DAQmx_Val_Volts, None)
        self.DAQ_signal.CreateAIVoltageChan(
            "Dev2/ai0", "", DAQmx_Val_Cfg_Default,
            -5.0, 5.0, DAQmx_Val_Volts, None)
        self.Triger.CreateCOPulseChanTime(
            "Dev2/ctr1", "triger", DAQmx_Val_Seconds,
            DAQmx_Val_Low, 0, 0.0005, 0.0005)

        self.DAQ_drive.CfgSampClkTiming(
            "", self.Rate, DAQmx_Val_Rising,
            DAQmx_Val_ContSamps, self.SampsN * 2)
        # triangle wave up and down, down time we do not take signal

        self.DAQ_signal.CfgSampClkTiming(
            "/Dev2/ao/SampleClock", self.Rate,
            DAQmx_Val_Rising, DAQmx_Val_ContSamps,
            self.SampsN * 2)
        # we just take signal at half period of drive! (triangle  rising time)

        self.DAQ_drive.CfgDigEdgeStartTrig(
            "PFI13", DAQmx_Val_Rising)  # crt1 ouput triger is PFI13
        self.DAQ_signal.CfgDigEdgeStartTrig("PFI13", DAQmx_Val_Rising)

        # 6 Ghz, four FSR @ 369 nm
        self.xscale = np.arange(self.SampsN)

    def SetWaveform(self, waveform):
        read = int32()
        self.waveform = waveform
        self.DAQ_drive.WriteAnalogF64(
            self.SampsN * 2, False, 10.0,
            DAQmx_Val_GroupByScanNumber,
            waveform, byref(read), None)

    def ReadData(self):
        read = int32()
        self.DAQ_signal.ReadAnalogF64(
            self.SampsN, 10.0, DAQmx_Val_GroupByScanNumber,
            self.data, self.SampsN, byref(read), None)

    def StopTask(self):
        self.DAQ_drive.StopTask()
        self.DAQ_signal.StopTask()
        self.Triger.StopTask()

    def ClearTask(self):
        self.DAQ_drive.ClearTask()
        self.DAQ_signal.ClearTask()
        self.Triger.ClearTask()

    def StartTask(self):
        self.DAQ_drive.StartTask()
        self.DAQ_signal.StartTask()
        self.Triger.StartTask()

    def SetTempFile(self, filename, method='a'):
        self.TempFile = open(filename, method)

    def __del__(self):
        self.ClearTask()
        if self.TempFile:
            self.TempFile.close()


class DLPro:
    inst = None

    def __init__(self, com='COM1'):
        rm = visa.ResourceManager()
        self.inst = rm.open_resource(com)

    def set(self, command):
        s = "(param-set! '" + command + ")"
        self.inst.write(s)
        for i in range(100):
            if(self.inst.read().find(s) != -1):
                break
        if i == 99:
            raise IndexValueError()
        return self.inst.read()

    def get(self, command):
        # "(param-ref 'laser1:dl:pc:voltage-act)"
        s = "(param-ref '" + command + ")"
        self.inst.write(s)
        # print(self.inst.read()[0:-2])
        for i in range(100):
            if(self.inst.read().find(s) != -1):
                break
        if i == 99:
            raise IndexValueError()
        return self.inst.read()

    def __del__(self):
        self.inst.close()
#####################################


def FindPeak(x, data, window=10):
    x0 = np.argmax(data)
    x = x[x0 - window:x0 + window]
    y = data[x0 - window:x0 + window]
    x0 = np.sum(x * y / np.sum(y))
    return x0


def GenFP_SimulationWaveform(N, x_range=4 * np.pi,
                             finesse=3000, Amp=1.0,
                             phase=np.pi, offset=0.0):
    x = np.linspace(0, x_range, N) + phase
    y = Amp / (1.0 + finesse**2 * np.sin(x / 2)**2) + offset
    return y


def GenTriangleWaveform(N, Amp=2.0, offset=0):
    waveform = np.ones((N,), dtype=np.float64)
    waveform[0:N / 2] = np.linspace(-Amp / 2, Amp / 2, N / 2) + offset
    waveform[N / 2:] = np.linspace(Amp / 2, -Amp / 2, N / 2) + +offset
    return waveform


if __name__ == "__main__":
    DAQ = DAQ_main()  # creat DAQ task
    Laser369 = DLPro('COM1')
    # set simulation waveform for debug
    # DAQ.SetWaveform(GenFP_SimulationWaveform(DAQ.SampsN))
    # set Triangle waveform for Scan
    waveform = GenTriangleWaveform(DAQ.SampsN * 2, Amp=5.0, offset=3.0)
    DAQ.SetWaveform(waveform)
    # Set App GUI
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 600)
    win.setWindowTitle('PyFP Scan')
    # Create docks, place them into the window one at a time.
    # give this dock the minimum possible size
    d1 = Dock("Setting", size=(1, 1))
    d2 = Dock("Scan", size=(500, 300))
    d3 = Dock("Peak to lock", size=(250, 300))
    d4 = Dock("Reference peak", size=(250, 300))
    d5 = Dock("Ploting", size=(500, 300))
    # area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it
    # will fill the whole space since there are no other docks yet)
    area.addDock(d1, 'top')
    area.addDock(d2, 'top')  # place d2 at right edge of dock area
    area.addDock(d5, 'top')  # place d2 at right edge of dock area
    area.moveDock(d2, 'bottom', d1)
    area.moveDock(d5, 'bottom', d1)
    area.moveDock(d2, 'above', d5)
    area.addDock(d3, 'bottom')
    area.addDock(d4, 'bottom')
    area.moveDock(d4, 'left', d3)  # move d2 to top edge of d3

    # Setting window
    w1 = pg.LayoutWidget()
    d1.addWidget(w1)
    ChkTrackPeak = QtGui.QCheckBox('TrackPeak    ')
    ChkTrackPeak.setChecked(False)
    LockInput = pg.LayoutWidget()
    LockInput.addWidget(ChkTrackPeak, row=0, col=0)
    labelRefPeakInput = QtGui.QLabel('RefDelta')
    LockInput.addWidget(labelRefPeakInput, row=0, col=1)
    SpinDetaV = pg.SpinBox(value=500, bounds=[-3000, 3000],
                           minStep=0.1, step=1, suffix='  MHz', decimals=5)
    LockInput.addWidget(SpinDetaV, row=0, col=2)
    w1.addWidget(LockInput, row=0, col=3)

    BtnLock = QtGui.QPushButton('Lock')
    BtnLock.setCheckable(True)
    BtnLock.setChecked(False)
    w1.addWidget(BtnLock, row=0, col=5)

    # Plot window
    w2 = pg.PlotWidget(title="Scanning")
    curve = w2.plot(pen='r')
    w2.setLabel('left', 'Amp', units='V')
    w2.setLabel('bottom', 'Frequency', units='MHz')
    x0, x1 = (DAQ.xscale[-1], DAQ.xscale[0])
    lr = pg.LinearRegionItem([x0, x0 + (x1 - x0) / 50])
    lr.setZValue(-10)
    lrr = pg.LinearRegionItem([x1 / 2, x1 / 2 - (x1 - x0) / 50])
    lrr.setZValue(-10)
    w2.addItem(lr)
    w2.addItem(lrr)
    d2.addWidget(w2)

    # Plot window 2
    w3 = pg.PlotWidget(title="Peak to Lock")
    curvePeak = w3.plot(pen='y')
    w3.setLabel('left', 'Amp', units='V')
    w3.setLabel('bottom', 'v', units='MHz')
    d3.addWidget(w3)

    w4 = pg.PlotWidget(title="Referece Peak")
    curvePeak2 = w4.plot(pen='r')
    w4.setLabel('left', 'Amp', units='V')
    w4.setLabel('bottom', 'v', units='MHz')
    d4.addWidget(w4)

      # Plot window
    w5 = pg.PlotWidget(title="Error Plot")
    curvePlot = w5.plot(pen='y')
    w5.setLabel('left', 'error', units='MHz')
    d5.addWidget(w5)

    PeakScaleMax = 1
    PeakScaleMin = 0
    PeakScaleMax2 = 1
    PeakScaleMin2 = 0

    def updatePlot():
        global PeakScaleMax, PeakScaleMin
        rag = lr.getRegion()
        w3.setXRange(*rag, padding=0)
        min = np.where(DAQ.xscale >= rag[0])
        max = np.where(DAQ.xscale <= rag[1])
        PeakScaleMax = np.max(max)
        PeakScaleMin = np.min(min)

    def updateRegion():
        global PeakScaleMax, PeakScaleMin
        lr.setRegion(w3.getViewBox().viewRange()[0])
        rag = lr.getRegion()
        min = np.where(DAQ.xscale >= rag[0])
        max = np.where(DAQ.xscale <= rag[1])
        PeakScaleMax = np.max(max)
        PeakScaleMin = np.min(min)

    def updatePlot2():
        global PeakScaleMax2, PeakScaleMin2
        rag = lrr.getRegion()
        w4.setXRange(*rag, padding=0)
        miny = np.where(DAQ.xscale >= rag[0])
        maxy = np.where(DAQ.xscale <= rag[1])
        PeakScaleMax2 = np.max(maxy)
        PeakScaleMin2 = np.min(miny)

    def updateRegion2():
        global PeakScaleMax2, PeakScaleMin2
        lrr.setRegion(w4.getViewBox().viewRange()[0])
        rag = lrr.getRegion()
        miny = np.where(DAQ.xscale >= rag[0])
        maxy = np.where(DAQ.xscale <= rag[1])
        PeakScaleMax2 = np.max(maxy)
        PeakScaleMin2 = np.min(miny)

    lr.sigRegionChanged.connect(updatePlot)
    w3.sigXRangeChanged.connect(updateRegion)

    lrr.sigRegionChanged.connect(updatePlot2)
    w4.sigXRangeChanged.connect(updateRegion2)
    updatePlot()
    updatePlot2()

    lastTime = time()
    fps = None

    cv = None  # Piezo Volt for current time
    errorData = np.empty(1000)
    errorData[:] = np.NAN

    def LockPeak(dx, P=1.0 / 1247 / 20, lowerLimit=57, upperLimit=61):
        global cv
        if cv is None:
            ss = Laser369.get('laser1:dl:pc:voltage-set')
            cv = float(ss)

        cv += P * dx
        print (cv, dx)

        if cv < lowerLimit:
            cv = lowerLimit
        if cv > upperLimit:
            cv = upperLimit
        Laser369.set('laser1:dl:pc:voltage-set %.6f' % cv)

    def update():
        global lastTime, fps, errorData
        global PeakScaleMin2, PeakScaleMax2, PeakScaleMin, PeakScaleMax

        DAQ.ReadData()
        curve.setData(y=DAQ.data,x=DAQ.xscale)
        yy = np.max(DAQ.data.reshape(2000, DAQ.SampsN / 2000), 1)
        xx = np.average(DAQ.xscale.reshape(2000, DAQ.SampsN / 2000), 1)
        #print DAQ.xscale,DAQ.data
        #curve.setData(DAQ.xscale, DAQ.data)
        app.processEvents()

        peaky = DAQ.data[PeakScaleMin:PeakScaleMax]
        peakx = DAQ.xscale[PeakScaleMin:PeakScaleMax]
        peaky2 = DAQ.data[PeakScaleMin2:PeakScaleMax2]
        peakx2 = DAQ.xscale[PeakScaleMin2:PeakScaleMax2]

        if BtnLock.isChecked():

            mp2 = (PeakScaleMax2 - PeakScaleMin2) / 2
            mp1 = (PeakScaleMax - PeakScaleMin) / 2

            PeakScaleMin += np.argmax(peaky) - mp1
            PeakScaleMax += np.argmax(peaky) - mp1
            PeakScaleMin2 += np.argmax(peaky2) - mp2
            PeakScaleMax2 += np.argmax(peaky2) - mp2

            lr.setRegion((DAQ.xscale[PeakScaleMin],
                          DAQ.xscale[PeakScaleMax]))
            lrr.setRegion((DAQ.xscale[PeakScaleMin2],
                           DAQ.xscale[PeakScaleMax2]))

            x1 = FindPeak(peakx, peaky)
            x2 = FindPeak(peakx2, peaky2)
            dx = x1 - x2 - SpinDetaV.value()
            # print('values',x1,x2,x2-x1,dx)
            LockPeak(dx)
            errorData = np.roll(errorData, 1)
            errorData[0] = dx
            curvePlot.setData(errorData)

        curvePeak.setData(y=peaky, x=peakx)
        curvePeak2.setData(y=peaky2, x=peakx2)
        app.processEvents()

        now = time()
        dt = now - lastTime
        lastTime = now
        if fps is None:
            fps = 1.0 / dt
        else:
            s = np.clip(dt * 3., 0, 1)
            fps = fps * (1 - s) + (1.0 / dt) * s

        w2.setTitle('Scanning, fps:%0.2f' % (fps))
        app.processEvents()

        DAQ.ReadData()  # Read the dowing data, it useless

    timer = QtCore.QTimer()
    timer.timeout.connect(update)

    DAQ.StartTask()
    timer.start(0.05)

    win.show()
    sys.exit(app.exec_())
    DAQ.StopTask()
