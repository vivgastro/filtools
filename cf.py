#--------------------------------------------------------------------------------------------
#       Author: VG
#       Code to quickly visualise dedispersed, time and frequency scrunched filterbank files
#--------------------------------------------------------------------------------------------

import numpy as N
import matplotlib.pyplot as M
from sigpyproc.readers import FilReader as F
import argparse
import os
import sys


F1_freq   = [840.0, 844.5]
F2_freq   = [835.5, 840.0]
F3_freq   = [830.0, 835.0]
Voda_freq = [825.25, 829.75]

def find_channels(freqs, edges):
    chw = freqs[1] - freqs[0]
    flipped = False
    if chw < 0:
        assert edges[1] - edges[0] > 0, "Stupid VG"
        freqs = freqs[::-1]
        flipped = True
    lower_edge = N.searchsorted(freqs, edges[0]) - 1
    upper_edge = N.searchsorted(freqs, edges[1])

    if flipped:
        lower_chan_edge = freqs.size-1 - upper_edge
        upper_chan_edge = freqs.size-1 - lower_edge
    else:
        lower_chan_edge = lower_edge
        upper_chan_edge = upper_edge


    lower_chan_edge = N.max([lower_chan_edge, 0])
    upper_chan_edge = N.min([upper_chan_edge, freqs.size])
    
    return lower_chan_edge, upper_chan_edge


def tscrunch(fsdata, tx):
    if tx==1:
        return fsdata
    
    nr=fsdata.shape[0]
    nc=fsdata.shape[1]
    
    endpoint=int(nc/tx) * tx
    
    tmp=fsdata[:,:endpoint].reshape(nr, nc//tx, tx)  #P.S.: We lose a few time samples in the end if the tx factor does not exactly divide the initial number of time samples in the filterbank file
    tfsdata=tmp.mean(axis=-1)
    
    return tfsdata
    
def fscrunch(data, fx):
    if fx==1:
        return data
    
    if data.shape[0]%fx!= 0:
        print("!!!No of freq channels must be an integer multiple of the Freq scrunch factor!!!")
        sys.exit(1)
    
    fsdata=N.zeros((data.shape[0]//fx, data.shape[1]))
    
    for i in range(data.shape[0]):
        fsdata[i//fx] += data[i]
    
    fsdata/=fx
    return fsdata


def parse_zap_chans(zapc, nch):
    zapc = zapc.strip(", \n\t").split(",")
    chans_to_zap = []
    for cc in zapc:
        cr = cc.strip().split(":")
        if len(cr) == 1:
            chans_to_zap.extend([int(cc.strip())])
            continue
        if len(cr)!=2:
            raise ValueError("Invalid channel range : {0}".format(cr))
        c1 = cr[0].strip()
        c2 = cr[1].strip()
        
        c1 = 0 if c1=='' else int(c1)
        c2 = nch-1 if c2=='' else int(c2)

        sign = N.sign(c2-c1)
        cs = N.arange(c1, c2+sign, sign).tolist()
        chans_to_zap.extend(cs)
    return N.unique(chans_to_zap).tolist()

def f_to_c(ff, freqs):
    chw = freqs[1] - freqs[0]
    f0_edge = freqs[0] - chw/2

    flow = N.min(freqs) - N.abs(chw)/2.
    fhigh = N.max(freqs) + N.abs(chw)/2.
    if ff < flow:
        raise ValueError("zap freq request ({0}) below valid range ({1})".format(ff, [flow, fhigh]))
    if ff > fhigh:
        raise ValueError("zap freq request ({0}) above valid range ({1})".format(ff, [flow, fhigh]))

    chan = int(  (ff - f0_edge) / chw  )
    return chan


def parse_zap_freqs(zapf, freqs):
    zapf = zapf.strip(", \n\t").split(",")
    chans_to_zap = []
    for ff in zapf:
        fr = ff.strip().split(":")
        if len(fr)==1:
            chans_to_zap.extend([f_to_c(float(ff.strip()), freqs)])
            continue
        if len(fr)!=2:
            raise ValueError("Invalid freq range : {0}".format(fr))
        f1 = fr[0].strip()
        f2 = fr[1].strip()

        c1 = f_to_c(N.min(freqs), freqs) if f1=='' else f_to_c(float(f1), freqs)
        c2 = f_to_c(N.max(freqs), freqs) if f2=='' else f_to_c(float(f2), freqs)

        sign = N.sign(c2-c1)
        cs = N.arange(c1, c2+sign, sign).tolist()
        chans_to_zap.extend(cs)
    return N.unique(chans_to_zap).tolist()
        
        

def parse_pols_args(nifs):
    if args.pol.strip().lower() == "all":
        pols_to_plot = N.arange(nifs)
    else:
        pp = args.pol.strip().split(",")
        try:
            pols_to_plot = [int(i.strip()) for i in pp]
        except ValueError as e:
            raise "Invalid polarisation options provided : {}".format(e)
        for pol in pols_to_plot:
            if pol > nifs-1:
                raise ValueError("Desired polarisation index {0} (starting from 0) > total no. of polarisations {1}".format(pol, nifs))
    return pols_to_plot


def split_pols(data, f):
    npols = f.header.nifs
    dpols = data.reshape(data.shape[0], -1, npols)
    split_data = [dpols[:, :, i] for i in range(npols)]
    return split_data

def rescale(data):
    rms = data.std(axis=-1)
    mean = data.mean(axis=-1)
    return (data-mean[:, None])/rms[:, None]


def main(args):
    print("Plotting {0} filterbank(s)".format(len(args.fil)))
    for k,fil in enumerate(args.fil):
        if not os.path.isfile(fil):
            print("{0} does not exist".format(fil))
            continue
        if not fil.endswith(".fil"):
            print("{0} is not a filterbank".format(fil))
            continue
        f=F(fil)
        tres=f.header.tsamp
        f0 = f.header.ftop      #It is not the highest freq always, but is = fch1 - foff/2
        fn = f.header.fbottom
        nch = f.header.nchans
        #if f0-fn > 0:
        #    (fn, f0) = (f0, fn)     #This ensures that f0 is the bottom frequency and fn is the top frequency always
        chw = f.header.foff
        
        #freqs = N.arange(f.header.fch1, f.header.fch1 + chw * nch, chw) 
        freqs = N.linspace(f.header.fch1, f.header.fch1 + f.header.foff * (f.header.nchans - 1) , f.header.nchans)

        if args.start > f.header.tobs:
            print("Sorry, start time: {0} > tobs: {1}".format(args.start, f.header.tobs))
            continue
        if args.ss and args.ss > f.header.nsamples:
            print("Sorry, start sample > nsamples")
            continue

        start=int(args.start/tres)
        if args.ss:
            start = args.ss
        start_time = start * f.header.tsamp

        #zero_level = 2**(f.header.nbits - 1)         #This is supposed to be the mean noise level. It is different if filterbanks are not 8 bit ones. 
        #zapped_level = zero_level - int(zero_level/100) #This is just to make the zapped plots look better. This does not affect the statistics at all.
        nsamp=int(args.duration/tres)
       
        if args.duration==-1:
            nsamp = f.header.nsamples 
        if args.nsamp:
            if args.nsamp == -1:
                nsamp = f.header.nsamples
            else:
                nsamp=args.nsamp

        if start+nsamp > f.header.nsamples:
            nsamp=f.header.nsamples - start

        #if f.header.nifs > 1:
        if args.pol is None:
            print("Polarisation not specified, plotting the 0th pol by default")
            pols_to_plot = [0]
        else:
            pols_to_plot = parse_pols_args(f.header.nifs)
        
        nsamp_times_nifs = nsamp * f.header.nifs

        data=f.read_block(start,nsamp_times_nifs)
        all_pols = data.reshape(data.shape[0], -1, f.header.nifs)
        
        for iip, ipol in enumerate(pols_to_plot):
            print("Plotting pol {0}".format(ipol))

            pdata = all_pols[:, :, ipol]

            #print nsamp, data.shape
            if args.rescale:
                rdata = rescale(pdata)
            else:
                rdata = pdata

            if args.dedisp:
                ddata=rdata.dedisperse(args.dedisp)
            else:
                ddata=rdata
            
            #if ddata.shape[0]%args.freq_sc!=0:
            #    print "Frequency scrunch factor should exactly divide the no. of channels({0})".format(ddata.shape[0])
            #    continue
            #endpoint=ddata.shape[1]-ddata.shape[1]%args.t_sc
            #print ddata.shape[1], args.t_sc, ddata.shape[1]%args.t_sc, ddata.shape[1]/args.t_sc, endpoint
            #print ddata[:,:endpoint].shape
            #We skip last few time samples by going upto endpoint only, just to enable time scrunching.
            #tfsdata=ddata[:,:endpoint].downsample(tfactor=args.t_sc, ffactor=args.freq_sc)
            useful_channels=ddata.shape[0]
            zapped=0
            if args.nuke:

                if 'F1' in args.nuke:
                    zapped = 1
                    z1, z2 = find_channels(freqs, F1_freq)
                    ddata[z1:z2, :] *= 0
                    useful_channels-=(z2-z1)

                if 'F2' in args.nuke:
                    zapped = 1
                    z1, z2 = find_channels(freqs, F2_freq)
                    ddata[z1:z2, :] *= 0
                    useful_channels-=(z2-z1)

                if 'F3' in args.nuke:
                    zapped = 1
                    z1, z2 = find_channels(freqs, F3_freq)
                    ddata[z1:z2, :] *= 0
                    useful_channels-=(z2-z1)

                if 'Voda' in args.nuke:
                    zapped = 1
                    z1, z2 = find_channels(freqs, Voda_freq)
                    ddata[z1:z2, :] *= 0
                    useful_channels-=(z2-z1)
       
            nch = len(freqs)
            if args.zapc:
                zapped=1
                zap_chans = parse_zap_chans(args.zapc, nch)
                if N.any(N.abs(zap_chans) >= nch):
                    raise ValueError("zap chan request out of bounds")
                ddata[zap_chans, :] *= 0
                useful_channels -= len(zap_chans)
            if args.zapf:
                zapped=1
                zap_chans = parse_zap_freqs(args.zapf, freqs)
                ddata[zap_chans, :] *= 0
                useful_channels -= len(zap_chans)
                

            fsdata=fscrunch(ddata,args.freq_sc)
            tfsdata=tscrunch(fsdata, args.t_sc)
            tseries=tfsdata.sum(axis=0)*1.0
            fseries = fsdata.sum(axis=1)*1.0
            fseries /= N.max(fseries)
            tseries/=useful_channels/args.freq_sc

            zero_level = N.mean(tseries)
            print(fil, "Mean:", zero_level, "Std:", N.std(tseries))

            zapped_level = zero_level * 0.95

            toff=0.5*tres*args.t_sc
            toff_samps = 0.5 * args.t_sc

            x=N.arange(0,len(tseries))*tres*args.t_sc + toff + start_time
            x_samps = N.arange(0, len(tseries), args.t_sc) + 0.5 * args.t_sc + start
            
            scrunched_freqs = N.linspace(f.header.ftop + chw/2 * args.freq_sc, f.header.fbottom - chw / 2 * args.freq_sc, f.header.nchans // args.freq_sc)
            #fa = f0 + chw/2*args.freq_sc
            #fb = fn - chw/2*args.freq_sc
            #scrunched_freqs = N.arange(fa, fb + chw*args.freq_sc, chw*args.freq_sc)
            #print(f"------> shape of scrunched_freqs = {scrunched_freqs.shape}, fa = {fa}, fb = {fb}, chw= {chw}, freq_sc = {args.freq_sc}")
            scrunched_chans = N.arange(0, nch, args.freq_sc) + 0.5 * args.freq_sc

            extent=[x[0]-toff, x[-1]+toff, fn, f0]
            extent_samps = [x_samps[0] - toff_samps, x_samps[-1] + toff_samps, nch + 1, 0]
            
            pk = k * len(pols_to_plot) + iip
            fig=M.figure(pk, figsize=(6.5,5))
            
            ax1=M.subplot2grid((6,8), (0,0), rowspan=4, colspan=6)
            zdata=tfsdata
            if zapped:
                zapped_chans = N.where(fseries < 1e-8)[0]
                zdata[zapped_chans,:]+=zapped_level
            if not args.raw_units:
                ax1.imshow(zdata, interpolation='none', aspect='auto', cmap='afmhot', extent=extent)
                ax1.set_ylabel("Freq (MHz)")
            else:
                ax1.imshow(zdata, interpolation='none', aspect='auto', cmap='afmhot', extent=extent_samps)
                ax1.set_ylabel("Freq channels idx")
                
            ax1.set_title(fil+" De-DM: "+str(args.dedisp)+", pol:{}".format(ipol), fontsize=8)
            ax1.set_xlim(0,tfsdata.shape[1])

            ax2=M.subplot2grid((6,8), (4,0), rowspan=1, colspan=6, sharex=ax1)
            if not args.raw_units:
                ax2.plot(x, tseries, linewidth=0.5)
                ax2.set_xlim(x[0]-toff, x[-1]+toff)
                ax2.set_xlabel("Time (s)")
            else:
                ax2.plot(x_samps, tseries, linewidth=0.5)
                ax2.set_xlim(extent_samps[0], extent_samps[1])
                ax2.set_xlabel("Time (samps)")
                
            
            ax3 = M.subplot2grid((6,8), (0,6), rowspan = 4, colspan=2, sharey=ax1)
            if not args.raw_units:
                ax3.plot(fseries, scrunched_freqs, linewidth=0.5)
                ax3.set_ylim(fn, f0)
            else:
                ax3.plot(fseries, scrunched_chans, linewidth=0.5)
                ax3.set_ylim(nch+1, 0)

            ax3.set_xlabel("Power")
            ax3.tick_params(labelsize=8)

            M.subplots_adjust(hspace=0, wspace=0, bottom=0.0)
            M.setp(ax1.get_xticklabels(), visible=False)
            M.setp(ax3.get_yticklabels(), visible=False)
            #M.setp(ax2.get_yticklabels(), visible=False)

            if not args.one:
                mgr=M.get_current_fig_manager()
                #mgr.window.move((pk%3)*640, int(pk/3)*600)
       
            if args.pngs:
                print("saving",fil)
                M.savefig(str(k+1).zfill(3)+"p{0}.png".format(ipol), dpi=200)
                M.close('all')
                continue

            if(k<len(args.fil)-1):
                M.show(block=False)
                if args.one:
                    input("<Press Enter to see next plot>\n")
                    M.close('all')
    if not args.pngs:
        M.show()

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("fil", type=str, nargs='+', help="Filterbank files to plot")
    a.add_argument("-s","--start", type=float, help="Time in seconds to start plotting from (def:0)", default=0.0)
    a.add_argument("-ss", type=int, help="Start sample", default=None)
    a.add_argument("-dur","--duration", type=float, help="Length in seconds to plot (def: 1s; say -1 for whole file)", default=1)
    a.add_argument("-ns","--nsamp", type=int, help="No. of time samples to plot, -1 for all", default=None)
    a.add_argument("-p", "--pol", type=str, help="Polarisations to plot separated by ',' (e.g., -p 0,1,2). You can say 'all' for all pols. \
            By default, the 0th polarisation will be plotted.", default=None)
    a.add_argument("-r", "--rescale", action='store_true', help="Rescale the data to 0 mean and 1 rms (def=False)", default=False)
    a.add_argument("-dd", "--dedisp", type=float, help="Dedispersion DM val (def=0)")
    a.add_argument("-fs","--freq_sc", type=int, help="Freq scrunch factor (def=1)", default=1)
    a.add_argument("-ts","--t_sc", type=int, help="Time scrunch factor (def=1)", default=1)

    a.add_argument("-zapf", "--zapf", type=str, help="Zap frequencies (MHz). You can give an individual frequency (e.g. -zapf 845.0.)"+\
            "Or a list of frequencies delimetered by ',' (e.g. -zapf 820.2, 825.2, 830.2)"+\
            "And/or an inclusive range of frequencies separated by ':' (e.g. -zapf 820.0 : 830.0)")
    a.add_argument("-zapc", "--zapc", type=str, help="Zap channels. You can give an individual channel (e.g. -zapc 25)"+\
            "Or a list of channels delimetered by ',' (e.g. -zapc 100, 200, 400)"+\
            "And/or an inclusive range of channels  separated by ':' (e.g. -zapc 100: 200)")
    a.add_argument("-nuke", "--nuke", type=str, nargs='+', help="nuke F1/F2/F3/Voda?")
    a.add_argument("--raw-units", action='store_true', help="Plot raw-units instead of physical units (def: False)", default=False)
    
    a.add_argument("-pngs", action='store_true', help="Save pngs instead of plotting")
    a.add_argument("-one", action='store_true', help="Show plots one by one (def=False)", default=False)
    args=a.parse_args()
    main(args)



