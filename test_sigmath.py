import unittest
from sigmath import *
from qamwrapper import QAMWrapper, QAMLayer
from stackcall import StackCall
import os
import matplotlib

## Run this to show an example of all plot tools
class plot(unittest.TestCase):

    def setUp(self):
        pass

    def getQam(self):
        samples = (2 ** 9) * 4 + 2
        msg_bits = np.random.randint(0, 2, samples)
        custom = QAMWrapper(64)  # This uses the default constructor
        modrf = custom.mod(msg_bits)
        return modrf

    def getWave(self):

        samples = int(1E3)
        fsdown = int(12E3)
        hz = 1E3
        data_orig = tone_gen(samples, fsdown, hz)
        return (data_orig, fsdown)

    def getRfUpTuple(self):
        data_orig, fsdown = self.getWave()

        # modup = interp6(data_orig)
        # fsup = fsdown*6
        return (data_orig, fsdown)




    def testNplotSpy(self):
        x = 6
        y = 45

        H = np.zeros((x, y))

        for i in range(x):
            for j in range(y):
                H[i,j] = np.random.random(1)>0.5

        fig = nplotspy(H, s_("random", x, ",", y, "size"))
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testDots(self):
        (rf, fs) = self.getWave()

        fig = nplotdots(rf[0:800], "Dots real time domain")
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testQam(self):
        (rf, fs) = self.getRfUpTuple()
        qam = self.getQam()

        fig = nplotqam(qam, "Qam")
        self.assertTrue(type(fig) is matplotlib.figure.Figure)
        # nplotqam(interp6(qam)[9:], "qam up by 6")

    def tearDown(self):
        pass
        # print "teardown "


    def testFFT(self):
        (rf, fs) = self.getWave()

        extend = np.concatenate((rf,np.zeros(10000)))

        figold = nplotfftold(extend, "OLD style of fft pre, 2017")

        fig = nplotfft(extend, fs, "new style (still not semilogy)")
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testHist(self):

        bins = np.random.normal(0,1000,25000)

        fig = nplothist(bins, "25k points of np normal in (0,1000) into 55 bins", 55)
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testBer(self):

        bers = []
        ebn0s = []
        titles = []

        ber = [0.158368318809598, 0.130644488522829, 0.103759095953406, 0.0786496035251426, 0.0562819519765415,
               0.037506128358926, 0.0228784075610853, 0.0125008180407376, 0.00595386714777866, 0.00238829078093281,
               0.000772674815378444, 0.000190907774075993, 3.36272284196175e-05, 3.87210821552204e-06]
        ebn0 = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        title = "theoretical"

        # first set
        bers.append(ber)
        ebn0s.append(ebn0)
        titles.append(title)

        # second set
        bers.append([0.144, 0.116, 0.098, 0.076, 0.058, 0.035, 0.021, 0.009, 0.004, 0.002, 0.002, 0, 0, 0])
        ebn0s.append(ebn0)  # same as before
        titles.append("j random sim run")

        fig = nplotber(bers, ebn0s, titles, "Lookup table of theoretical BPSK Bit Error Rates")
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testRainbow(self):
        sz = 314*2
        arguments = range(0,sz)
        arguments = np.array(arguments)/44.0

        fs = int(2*np.pi*100)
        sinrf = np.exp(1j*arguments)

        nplot(np.imag(sinrf), "Hold On")
        plt.hold(True)
        for x in drange(-0.9, 1.9, 0.05):
            shift = tone_gen(sz, fs, x)
            shifted = shift * sinrf

            nplot(np.imag(shifted), newfig=False)


    def testMulti(self):
        liny = [1,2,3,4,5,6]
        blueline = [1E1, 1E2, 1E3, 1E4, 1E5, 1E6]
        greenline = [1,2,3,4,5,6]

        xvals = [blueline,greenline]
        yvals = [liny,liny]
        leg = ["Exponential (blue)", "Linear (green)"]
        xlabel = "text for normies"
        ylabel = "text the funny way"
        title = "semilogy view of exponential vs linear values"
        semilog = True

        fig1 = nplotmulti(xvals,yvals,leg,xlabel,ylabel,title,semilog)
        self.assertTrue(type(fig1) is matplotlib.figure.Figure)
        fig2 = nplotmulti(xvals, yvals, leg, xlabel, ylabel, "Linear view of exponential vs linear values", False)
        self.assertTrue(type(fig2) is matplotlib.figure.Figure)


    def testAngle(self):
        chunks = 9
        target = np.pi
        for i in range(chunks):
            rad = i*(target/(chunks-1))
            nplotangle(rad, s_("Theta [0,", target, "] in ", chunks, " lines "), False)

        fig = nplotangle(3*np.pi/4, "3pi/4")
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testResponse(self):
        fs = int(6E9)
        cutoff = 6.654321E7
        brx, arx = butter_lowpass(cutoff, fs, order=10)
        self.assertEqual(len(brx),len(arx))
        taps = len(brx)

        title = s_('Butterworth lowpass 10th order\ncutoff:',cutoff,'sample rate:', fs, 'taps:',taps)

        fig = nplotresponse(brx, arx, 'frequency', title, fs, cutoff)
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testFFTPeaks(self):
        (rf, fs) = self.getWave()

        extend = np.concatenate((rf, np.zeros(10000)))

        fig = nplotfft(extend, fs, "FFT with hz annotations at peaks", peaks=1, peaksHzSeparation=12)
        self.assertTrue(type(fig) is matplotlib.figure.Figure)

    def testPlotText(self):
        str = "A quick brown\nfox jumps\nover the lazy dog"


        names = ["ha", "sd", "ss", "xx"]
        for i in range(3):
            plt.subplot(2, 2, i + 1)
            nplotangle(1.1 * i, names[i], False)

        nplottext(str, plt.subplot(2,2,4))
        nplottext("free standing\ntext")



    @classmethod
    def tearDownClass(cls):
        print "Showing all figures"
        nplotshow()


class working(unittest.TestCase):

    def testTypicalXcr(self):
        a1 = np.exp(1j*3.0)
        a2 = np.exp(1j*2.8)
        s1 = [a1]
        packet =  np.array([0, a2, 0])

        # hard way
        xcr = np.correlate(packet, s1, 'full')
        absxcr = abs(xcr)
        xcrmax, xcridx = sig_max(absxcr)
        peaksample = xcr[xcridx]
        packetangle = np.angle(peaksample)


        expected = np.angle(a2)-np.angle(a1)

        # sanity check
        self.assertAlmostEqual(packetangle, expected, 7, "Sanity check failed, one of broke: math, sig_max(), np.correlate()")

        # easy way
        idx, xcrpk, ang = typical_xcorr(packet, s1)

        self.assertAlmostEqual(ang, expected, 7, "typical_xcorr failed to get angle correct")
        self.assertAlmostEqual(idx, 1, 7, "typical_xcorr failed to get idx correct")

    def testComplexRawMultiFile(self):
        samples = tone_gen(100, 100, 14.5)

        f1 = '/tmp/tcrm1.raw'
        f2 = '/tmp/tcrm2.raw'
        f3 = '/tmp/tcrm3.raw'

        save_rf_grc(f1, samples)


        dumpfile = open(f2, 'w')
        for s in samples:
            dumpfile.write(complex_to_raw(s))
        dumpfile.close()


        dumpfile3 = open(f3, 'w')
        ms = complex_to_raw_multi(samples)
        dumpfile3.write(ms)
        dumpfile3.close()

    def testComplexRawMultiSimple(self):
        samples = tone_gen(2, 100, 14.5)

        o1 = ''

        for s in samples:
            o1 += complex_to_raw(s)

        # print_rose(o1)

        o2 = complex_to_raw_multi(samples)

        # print_rose(o2)

        self.assertEqual(o1, o2, "complex_to_raw_multi is wrong")




        pass

    def testSaveGrc(self):
        samples = np.arange(0, 1, 0.1, dtype=np.complex128)


        fname = '/tmp/sigmathtest1.raw'

        save_rf_grc(fname, samples)

        readback = read_rf_grc(fname)

        self.assertTrue(np.allclose(readback, samples))

        readback2 = read_rf_grc(fname, 4)

        self.assertTrue(np.allclose(readback2, samples[0:4]))

        readback3 = read_rf_grc(fname, 40) # too long
        self.assertTrue(np.allclose(readback3, samples))


    ## \test Test if unique_matrix_pair works and if it respects upper triangular flag
    def testUniquePair(self):
        pairs = unique_matrix_pair(3)

        expected = [(0,1),(0,2),(1,2)]

        for e in expected:
            self.assertTrue(e in pairs, "unique_matrix_pair(3) not correct")
        # print pairs

        # for x in pairs:
        #     print x

        pairslt = unique_matrix_pair(3, True)

        self.assertNotEqual(pairs, pairslt, "unique_matrix_pair() not respecting lower triangular flag")

        for x in pairslt:
            # print x
            p1 = x[0]
            p2 = x[1]

            self.assertTrue((p2,p1) in pairs, "lt not found in ut version of unique_matrix_pair()")



    ## \test
    # Test if sigfdelay of 0.0 and 1.0 are valid, does not test center
    def testSigfdelay(self):

        signal = np.array([0.0, 1.0, 0.75, 0.5, 0.25, 0.0])

        res1 = sigfdelay(signal, 0.0)
        self.assertEqual(list(signal), list(res1), "fdelay of 0.0 is not idential to input")


        res2 = sigfdelay(signal, 1.0)
        res2_expected = np.concatenate(([0],res2[1:]))
        self.assertEqual(list(res2), list(res2_expected), "fdelay of 1.0 is not identical to input shifted by 1 sample")

        # indices = range(len(signal))
        # nplotmulti([signal, res2], [indices, indices], ['Orig', 'resampled'], 'samples in', 'samples out', title='resampled', newfig=True)
        # nplotshow()



    ## \test tone_gen() should work for fractional hz or something
    def testToneGen(self):
        count = int(12E3)
        fs = int(26E6)
        res = tone_gen(count, fs, 06.0E6 + 1 * 5000)

        # seems like a trivial test, but previous tone_gen fails this
        self.assertEqual(len(res), count)

    ## \test Test if sig_lin_interp() works
    def test_lin_interp_baked(self):
        self.assertEquals(sig_lin_interp(1, 2, 1.0), 2.0)
        self.assertEquals(sig_lin_interp(0, 100, 0.5), 50.)

    ## \test Test, circle_shift_peak() should always circle shifts correctly
    def test_phase_recovery(self):
        # V is right most sample
        t1 = [3972.0134951819764, 5743.9865218806526, 6641.4896699904775, 6527.8862210980451, 5420.4712704794019,
              3487.8387056246884, 1024.2143151752171, 1595.3374202337948]
        # V is center
        t2 = [6452.3443262904584, 5242.6348373616111, 3234.7817190476057, 734.46240737790356, 1877.6721478967816,
              4203.9481397945883, 5890.2111362934529, 6679.7428821885542]
        # V is right of center
        t3 = [6419.8612069240753, 6691.1477874040193, 5943.7677726574584, 4291.5029949128266, 1985.8957887642405,
              622.04604903354812, 3135.2870147274548, 5171.2089538766695]
        # V is flat, and max is split across RL boundary
        t4 = [5546.2553321930354, 3672.018754587069, 1238.7506085281454, 1383.106088376979, 3794.3974212146222,
              5628.0261429706143, 6604.8389026433606, 6576.1248124024023]

        t1ideal = np.roll(t1,1)


        expectedmax = [3,3,4,3]

        for i in range(4):
            series = [t1, t2, t3, t4][i]
            (rolled, shift) = circle_shift_peak(series)
            (maxval,maxidx) = sig_max(rolled)
            assert maxidx == expectedmax[i]

        for i in range(8):
            t1shift = np.roll(t1, i)
            (rolled, shift) = circle_shift_peak(t1shift)
            self.assertItemsEqual(t1ideal, rolled)

    ## \test Test polyfit_peak_climb(), bakes in some values
    def test_interp_peak_climb(self):
        t1 = [3972.0134951819764, 5743.9865218806526, 6641.4896699904775, 6527.8862210980451, 5420.4712704794019,
              3487.8387056246884, 1024.2143151752171, 1595.3374202337948]
        t2 = [6452.3443262904584, 5242.6348373616111, 3234.7817190476057, 734.46240737790356, 1877.6721478967816,
              4203.9481397945883, 5890.2111362934529, 6679.7428821885542]
        t3 = [6419.8612069240753, 6691.1477874040193, 5943.7677726574584, 4291.5029949128266, 1985.8957887642405,
              622.04604903354812, 3135.2870147274548, 5171.2089538766695]
        t4 = [5546.2553321930354, 3672.018754587069, 1238.7506085281454, 1383.106088376979, 3794.3974212146222,
              5628.0261429706143, 6604.8389026433606, 6576.1248124024023]

        showplot = False
        order = 4

        expected = [3.39004538, 3.2796452, 3.76305884, 3.47221838]

        for i in range(4):
            series = [t1, t2, t3, t4][i]
            (rolled, shift) = circle_shift_peak(series)
            # nplotfigure()
            max_x = polyfit_peak_climb(rolled, order, showplot)

            # self.assertAlmostEqual(max_x, expected[i], 4)
            # print "max_x was", max_x

        if showplot:
            nplotshow()

    ## \test Test tone_gen against Matlab example
    def test_tone_gen(self):

        # matlab output for: freq_shift(ones(1,20),10,1)
        expected = [  1.0000 + 0.0000j,
           0.8090 + 0.5878j,
           0.3090 + 0.9511j,
          -0.3090 + 0.9511j,
          -0.8090 + 0.5878j,
          -1.0000 + 0.0000j,
          -0.8090 - 0.5878j,
          -0.3090 - 0.9511j,
           0.3090 - 0.9511j,
           0.8090 - 0.5878j,
           1.0000 - 0.0000j,
           0.8090 + 0.5878j,
           0.3090 + 0.9511j,
          -0.3090 + 0.9511j,
          -0.8090 + 0.5878j,
          -1.0000 + 0.0000j,
          -0.8090 - 0.5878j,
          -0.3090 - 0.9511j,
           0.3090 - 0.9511j,
           0.8090 - 0.5878j]


        generated = tone_gen(20,10,1).tolist()

        self.assertEqual(len(generated), len(expected))

        places = 3
        for i in range(len(generated)):
            self.assertAlmostEqual(np.real(expected[i]), np.real(generated[i]), places)
            self.assertAlmostEqual(np.imag(expected[i]), np.imag(generated[i]), places)

    ## \test Test that sig_fft works
    def testSigFft(self):
        arguments = np.array(range(0,314*2))*2.0

        fs = 2*np.pi*100 # aka 628
        swave = np.exp(1j*arguments)

        (fdata, freqs) = sig_fft(swave, fs)

        absdata = abs(fdata)

        m,idx = sig_max(absdata)
        hz = freqs[idx]

        self.assertAlmostEqual(200, hz, 0)

    ## \test Test that sif_fft with zero padding works
    def testSigFftStretch(self):
        arguments = np.array(range(0,314*2))*2.0

        fs = 2*np.pi*100 # aka 628
        swave = np.exp(1j*arguments)

        swave = np.append(swave, [0]*10000)

        (fdata, freqs) = sig_fft(swave, fs)

        absdata = abs(fdata)

        m,idx = sig_max(absdata)
        hz = freqs[idx]

        self.assertAlmostEqual(200, hz, 3)

    ## \test Test that complex_to_raw() and raw_to_complex() convert correctly
    def testConversionsSingles(self):
        cnt = 1000

        rnd = np.random.random(cnt) + np.random.random(cnt)*1j

        bytes = ""

        for i in range(cnt):
            bytes = bytes + complex_to_raw(rnd[i])

        # print "made bytes:", len(bytes)

        self.assertEquals(len(bytes) % 8, 0)


        rndout = []

        for i in range(0, len(bytes), 8):
            gonuse = bytes[i:i+8]
            # print len(gonuse)
            # print repr(gonuse)
            rndout.append(raw_to_complex(gonuse))

        self.assertEquals(len(rnd), len(rndout))

        for i in range(len(rnd)):
            self.assertAlmostEqual(rnd[i], rndout[i], 6)

    ## \test Test that raw_to_complex_multi() works
    def testConversionsMulti(self):

        cnt = 1000

        rnd = np.random.random(cnt) + np.random.random(cnt) * 1j

        bytes = ""

        for i in range(cnt):
            bytes = bytes + complex_to_raw(rnd[i])

        # print "made bytes:", len(bytes)

        self.assertEquals(len(bytes) % 8, 0)

        rndout = raw_to_complex_multi(bytes)

        self.assertEquals(len(rnd), len(rndout))

        for i in range(len(rnd)):
            self.assertAlmostEqual(rnd[i], rndout[i], 6)

    # ## \test Test sig_sha256 wrapper and sig_sha256_matrix() (that marshal matrices into a standard format) work
    # def testSigSha(self):
    #     empty = sig_sha256("")

    #     # test basics, https://en.wikipedia.org/wiki/SHA-2
    #     self.assertEqual(empty, "\xe3\xb0\xc4\x42\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99\x6f\xb9\x24\x27\xae\x41\xe4\x64\x9b\x93\x4c\xa4\x95\x99\x1b\x78\x52\xb8\x55", "empty string not correct")

    #     H1 = np.eye(42, dtype=np.int16)

    #     sha1 = sig_sha256(np.int16(H1).tolist())
    #     sha2 = sig_sha256_matrix(H1)
    #     self.assertEqual(sha1, sha2, "sig_sha256_matrix not converting right")

    #     sha3 = sig_sha256_matrix(np.eye(42, dtype=np.float64))
    #     self.assertEqual(sha1, sha3, "not correct when eye() generated with float64")

    #     H2 = np.double(H1)
    #     sha4 = sig_sha256_matrix(H2)
    #     self.assertEqual(sha4, sha1, "failed after explicit cast to double")


    #     H1sparse = scipy.sparse.eye(42)
    #     sha5 = sig_sha256_sparse_matrix(H1sparse)
    #     self.assertEqual(sha5, sha1, "failed after eye was generated with sparse")

    #     H1sparseconv = scipy.sparse.bsr_matrix(H1)
    #     sha6 = sig_sha256_sparse_matrix(H1sparseconv)
    #     self.assertEqual(sha6, sha1, "failed after dense eye was made sparse")


    ## \test Test that get_rose() and reverse_rose() work
    def testRose(self):
        data = rand_string_ascii(15)

        rose = get_rose(data)

        reloaded = reverse_rose(rose)

        self.assertEqual(reloaded, data)


class PrintSyntaxSugar(unittest.TestCase):

    def testPathloss(self):
        fc = 2000000000
        distance = 2355.55
        ch_Type = 'a'
        htx = 10
        hrx = 2
        corr_fact = 'atnt'
        mod = 'mod'

        res = o_PL_IEEE80216d(fc,distance,ch_Type,htx,hrx,corr_fact,mod)

        # print "res",res

    def testSimpleSnr(self):
        fc = 915E6
        fc = 2E9
        distance = 1500
        distance = 184
        bw = 20E6
        mw = 1000

        snr = simple_snr(bw,mw,fc,distance)

        # print "snr", snr


    def test_basic(self):
        capture = StringIO.StringIO()
        print >>capture, 'Second line.'
        out = s_('Second line.')
        self.assertEqual(capture.getvalue()[:-1], out)

    def test_none(self):
        capture = StringIO.StringIO()
        print >>capture, None
        out = s_(None)
        self.assertEqual(capture.getvalue()[:-1], out)

    def test_afew(self):
        capture = StringIO.StringIO()
        print >>capture, 1, '2', 3
        out = s_(1, '2', 3)
        self.assertEqual(capture.getvalue()[:-1], out)

    def test_list(self):
        capture = StringIO.StringIO()
        print >>capture, [3,4,'a'], 'a', 3.14, capture
        out = s_([3,4,'a'], 'a', 3.14, capture)
        self.assertEqual(capture.getvalue()[:-1], out)

    def testDrange(self):

        d1 = []
        for x in drange(-0.2,0.2,0.1):
            d1.append(x)

        d2 = []
        for x in drange_DO_NOT_USE(-0.2,0.2,0.1):
            d2.append(x)

        d3 = []
        for x in drange_DO_NOT_USE2(-0.2,0.2,0.1):
            d3.append(x)

        builtin = []

        for x in [i/10. for i in range(-2,2)]:
            builtin.append(x)

        # print [i/10. for i in range(-2,2)]
        # print drange(-0.2,0.2,0.1)
        # print d3
        # print frange3(-0.2,0.2,0.1)

        # print d1
        # print d2
        # print d3
        # print builtin

        self.assertListEqual(builtin, d1, "drange does not match built in range")
        self.assertEqual(d1, builtin)

        # self.assertNotEqual(builtin, d2, "one I thought was bad actually was good")
        self.assertNotEqual(builtin, d3, "one I thought was bad actually was good")


    def tesxJustPrint(self):

        ranges = [[-0.2,0.2,0.1],[0, 0.1, 0.01]]

        for rr in ranges:
            # rr[0], rr[1], rr[2]

            print "starting on ", rr[0], rr[1], rr[2]
            print "#" * 15
            d1 = []
            for x in drange(rr[0], rr[1], rr[2]):
                d1.append(x)

            d2 = []
            for x in drange_DO_NOT_USE(rr[0], rr[1], rr[2]):
                d2.append(x)

            d3 = []
            for x in drange_DO_NOT_USE2(rr[0], rr[1], rr[2]):
                d3.append(x)

            d4 = []
            for x in np.arange(rr[0],rr[1],rr[2]):
                d4.append(x)

            print d1
            print d2
            print d3
            print d4
            print ""
            print ""

        #
        # self.assertListEqual(builtin, d1, "drange does not match built in range")
        # self.assertEqual(d1, builtin)
        #
        # # self.assertNotEqual(builtin, d2, "one I thought was bad actually was good")
        # self.assertNotEqual(builtin, d3, "one I thought was bad actually was good")


    def tesxInterpfft(self):
        # force octave server
        # sigmath_octave_use_client()

        arguments = range(0,314*2)
        arguments = np.array(arguments)*10.0

        sinrf = np.exp(1j*arguments*-1)

        nplotfft(sinrf)

        modup = interpn(sinrf, 3, 1, 0.333)

        nplotfft(modup)


        # nplot(np.imag(sinrf))
        # nplot(np.imag(sinrf))

        nplotshow()





    def tesxSigDiff(self):
        s = "hello"
        for i in range(120, 130):
            s += chr(i)
        sig_diff(s, s[0:8])

        s1 = ""
        for i in range(0, 256):
            s1 += chr(i)
        sig_diff(s1, s1[0:78])

    def tesxSigMax(self):
        count = 10000000

        big = abs(np.random.rand(count)*1j)
        big[700000] = 99
        biglist = list(big)

        a = time.time()
        mx, idx = sig_max(big)
        b = time.time()
        print "found max in", b-a
        self.assertEqual(idx, 700000)
        self.assertEqual(mx, 99)

        a = time.time()
        mx, idx = sig_max(biglist)
        b = time.time()
        print "found max in", b-a
        self.assertEqual(idx, 700000)
        self.assertEqual(mx, 99)

#        -----------------

        a = time.time()
        mx, idx = sig_max2(big)
        b = time.time()
        print "found max in", b-a
        self.assertEqual(idx, 700000)
        self.assertEqual(mx, 99)

        a = time.time()
        mx, idx = sig_max2(biglist)
        b = time.time()
        print "found max in", b-a
        self.assertEqual(idx, 700000)
        self.assertEqual(mx, 99)

#        -----------------

        # a = time.time()
        # mx, idx = sig_max3(big)
        # b = time.time()
        # print "found max in", b-a
        # self.assertEqual(idx, 700000)
        # self.assertEqual(mx, 99)

        a = time.time()
        mx, idx = sig_max3(biglist)
        b = time.time()
        print "found max in", b-a
        self.assertEqual(idx, 700000)
        self.assertEqual(mx, 99)

#        -----------------

        # a = time.time()
        # mx, idx = sig_max4(big)
        # b = time.time()
        # print "found max in", b-a
        # self.assertEqual(idx, 700000)
        # self.assertEqual(mx, 99)

        a = time.time()
        mx, idx = sig_max4(biglist)
        b = time.time()
        print "found max in", b-a
        self.assertEqual(idx, 700000)
        self.assertEqual(mx, 99)


    def tesxSigChannel(self):
        mod = QAMWrapper(64)

        ideal_bits = str_to_bits(rand_string_ascii(400))

        signal = mod.mod(ideal_bits)

        # snr = 6
        nplotqam(signal)
        for snr in range(15, 40, 5):
            noisy = sig_awgn(signal, snr)
            nplotqam(noisy,str(snr))


        nplotshow()

        pass


    # requries 'octave_server.py' to run in a separate process
    def testOctaveClient(self):
        c = get_octave_via_server()

        # push and pull an eye matrix
        c.eval('a = eye(3)')
        aout = c.pull('a')
        self.assertEqual(aout.tolist(), np.eye(3).tolist())



    def testSaveLoad(self):
        path = 'tmp/test_sigmath_t1.npz'

        try:
            os.remove(path)
        except OSError:
            pass

        H = np.eye(4200, dtype=np.int16)

        sha1 = sig_sha256_matrix(H)

        Hsparse = scipy.sparse.bsr_matrix(H, dtype=np.int16)

        save_sparse_csr(path, Hsparse)

        time.sleep(0.001)

        Hloaded = load_sparse_csr(path)

        self.assertEqual(sig_sha256_sparse_matrix(Hsparse), sig_sha256_sparse_matrix(Hloaded))


## some junk for sig_peaks()
class proto(unittest.TestCase):
    ## \test Test str_to_bits() and str_to_bits_cython() identical
    def test_str_to_bits_cython(self):
        s = rand_string_ascii(1000)

        # a = time.time()
        resold = str_to_bits(s)
        # print "ran in ", time.time()-a

        # b = time.time()
        res = str_to_bits_cython(s)
        # print "ran in ", time.time()-b

        self.assertEqual(list(res), list(resold))

    def testSave(self):
        data = [1+0j, 0.5+0.5j, 0.1, + 0.9j]
        nplotqam(data)
        plt.ylim([-1,1])
        plt.xlim([-1,1])
        plt.savefig('other.png')

        # if you generate a bunch of png's this way with a predictable filename
        # such as:
        #   filename = "filename%02d.png" % (i,)
        # you can run
        #   ffmpeg -i filename%02d.png -qmin 0 -qmax 1 -sws_dither bayer output.gif


        # nplotshow()

    def tesxSubplot(self):
        names = ["ha","sd","ss","xx"]
        for i in range(4):
            plt.subplot(2, 2, i+1)
            nplotangle(1.1*i, names[i], False)
            # names[i]
        nplotshow()

    def tesx_sigpeaks(self):
        fs = int(1E4)
        sz = 2*fs # should produce half hz bins
        tone = 3042.24
        t1 = tone_gen(sz, fs, tone)
        fig = nplotfft(t1, fs, peaks=2)

        bins,hz = sig_fft(t1,fs)


        # res = sig_peaks(bins, hz, 1, 1)
        # self.assertEqual(1,len(res))
        # self.assertAlmostEqual(hz[res[0]], tone, 0)

        res2 = sig_peaks(bins, hz, 2, 200)
        print res2

        #
        # print res


        #


        nplotshow()


    def tesx_sigpeaks(self):
        fs = int(1E4)
        sz = int(0.3*fs) # should produce half hz bins
        tonehz = 3042.24
        tone2hz = 1000
        d1 = tone_gen(sz, fs, tonehz) + tone_gen(sz, fs, tone2hz)

        d2 = np.concatenate((d1,np.zeros(1)))


        fig = nplotfft(d2, fs, peaks=2, peaksHzSeparation=2040)
        nplotshow()




if __name__ == '__main__':
    unittest.main()


