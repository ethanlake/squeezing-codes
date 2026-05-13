import numpy as np
import matplotlib

# Shim for matplotlib >= 3.9, which removed matplotlib.cbook._Stack that
# matplotlib.widgets.Slider still references in some versions.
try:
    import matplotlib.cbook
    if not hasattr(matplotlib.cbook, "_Stack"):
        class _Stack(list):
            def push(self, item):
                self.append(item)
                return item
            def pop(self):
                return super().pop() if self else None
            def current(self):
                return self[-1] if self else None
            def forward(self):
                pass
            def back(self):
                pass
        matplotlib.cbook._Stack = _Stack
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

# these need to first be defined outide of any function since we are treating them as global later 
nu = 1.; beta = 1/8.; gamma = 1.75; z = 1. # ising values 
# nu = 1.; beta = .2; gamma = 1.6 # s3 values 

def scaling_plotter(data,plot,xc,nu0=1.,gamma0=1.75,beta0=.125,z0=2.1667,raw=False,d=2,title="",cmap=cm.coolwarm,
                    fit_params=None, fit_uncert=None):
    """
    makes interactive scaling plots for various critical exponents.
    data: a dictionary with the following entries
        * "Ls": system sizes
        * "xs": raw error rates (not reduced to pc) for each system size (#Ls x #(x data points) matrix)
        * "chis": susceptibilities
        * "mags": magnetizations
        * "binds": binder cumulants
        * "t_autos": magnetization autocorrelation time (stats files only)
        * "t_rels":  first-passage / relaxation time (trel files only)
    plot: quantity to plot, one of ["binds", "mags", "chis", "t_autos", "t_rels"].
        `t_autos` and `t_rels` share an identical scaling form (τ ~ L^z)
        and use the same code path here; only axis labels differ.
    xc: value of critical point
    nu,gamma,beta: estimates for critical exponents
    raw: if true, just plots the raw data without scaling
    d: dimension; used only for checking hyperscaling relations
    title: plot title
    fit_params:  optional dict {pc, nu, beta, gamma, z} from an automated
        collapse fit. When supplied, the module-level tuning seeds are
        overridden with these values so keypress tuning starts at the fit.
    fit_uncert:  optional dict {name: (lower, upper)} 1σ bootstrap
        uncertainties; if given, the title annotates each fitted exponent as
        `ν = 0.92 (+0.05 / -0.04)` etc.
    critical exponents tuned using key presses. left/right tunes nu, right/left tunes gamma, and ,/. tunes beta
    """
    # Seed the module-level tuning variables with fit values when supplied,
    # so keypress increments start from the fit rather than the hard-coded
    # Ising defaults at the top of the file.
    global nu, beta, gamma, z
    if fit_params is not None:
        if "nu"    in fit_params: nu    = float(fit_params["nu"])
        if "beta"  in fit_params: beta  = float(fit_params["beta"])
        if "gamma" in fit_params: gamma = float(fit_params["gamma"])
        if "z"     in fit_params: z     = float(fit_params["z"])
        nu0    = nu
        beta0  = beta
        gamma0 = gamma
        z0     = z
    # cmap = cm.coolwarm 
    ### asthetics for plots ## 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']
    mew = 1.
    ms = 6.5 * 0.75 * 0.8
    lw = 2.2 #* 0
    a = .8

    fig,ax = plt.subplots(figsize=(2.5 * 1.25, 2.5 * 0.8))
    ax.minorticks_off()

    Ls = data["Ls"]; chis = data["chis"]; mags = data["mags"]; binds = data["binds"]
    # Helper: is `plot` the inverse-time scaling branch (either flavor)?
    _is_inv_time = lambda p: p in ("t_autos", "t_rels")

    def _update_errorbar(container, x, y, yerr):
        """
        Reposition an existing matplotlib errorbar container after data changes.
        `container` is the ErrorbarContainer returned by ax.errorbar(); may be
        None (e.g. when errors are unavailable).
        """
        if container is None:
            return
        _, caplines, barlinecols = container.lines[0], container.lines[1], container.lines[2]
        if yerr is None:
            # hide caps and vertical segments when no error info is available.
            for cap in caplines:
                cap.set_data([], [])
            for bars in barlinecols:
                bars.set_segments([])
            return
        x = np.asarray(x); y = np.asarray(y); yerr = np.asarray(yerr)
        lower = y - yerr
        upper = y + yerr
        # Rebuild the vertical line segments for the bars.
        segs = [np.array([[xi, lo], [xi, up]]) for xi, lo, up in zip(x, lower, upper)]
        for bars in barlinecols:
            bars.set_segments(segs)
        # Caps: typically [lower_caps, upper_caps].
        if len(caplines) >= 2:
            caplines[0].set_data(x, lower)
            caplines[1].set_data(x, upper)

    # error-bar keys corresponding to each observable key; absent or all-NaN
    # entries in the data dict fall back to NaN error (rendered as no bar).
    err_key = {"mags": "mags_err", "chis": "chis_err",
               "binds": "binds_err",
               "t_autos": "t_autos_err", "t_rels": "t_rels_err"}

    def _err(plot_key, l):
        k = err_key.get(plot_key)
        if k is None or k not in data:
            return None
        e = np.asarray(data[k][l], dtype=float)
        return None if np.all(np.isnan(e)) else e

    ### plot raw unscaled data ###
    if raw:
        for l in range(len(Ls)):
            col = cmap((l+1) / len(Ls))
            L = Ls[l]

            xs = data["xs"][l]; ts = (xs - xc)/xc
            ys = data[plot][l] # stuff to plot at this system size
            yerr = _err(plot, l)
            if _is_inv_time(plot):
                # plot inverse memory lifetime; propagate 1/y error by 1/y^2.
                if yerr is not None:
                    yerr = yerr / np.where(ys == 0, np.nan, ys)**2
                ys = L**0/ys
                ax.set_yscale('log'); ax.set_xscale('log')

            ax.errorbar(xs, ys, yerr=yerr, c=col, label=r'$%d$'%Ls[l],
                        marker='o', mec='k', mew=mew, ms=ms, lw=lw,
                        elinewidth=0.9, capsize=2)
        ax.set_xlabel(r'$p$')
        if plot == "mags":
            ax.set_ylabel(r'$M$')
        elif plot == "chis":
            ax.set_ylabel(r'$\chi$')
        elif plot == "binds":
            ax.set_ylabel(r'$B$')
        elif plot == "t_autos":
            ax.set_ylabel(r'$1/t_{\rm auto}$')
        elif plot == "t_rels":
            ax.set_ylabel(r'$1/t_{\rm rel}$')

        ax.legend(title=r'$L$',title_fontsize=12)
        ax.set_title(title)

    ### plot scaled data (in an interactive way) ###
    else:
        lines0 = [] # will hold the plot lines; dynamically updated
        ebars  = [] # corresponding error-bar containers (may be None)

        # draw lines with guessed values of critical exponents
        for l in range(len(Ls)):
            col = cmap((l+1) / len(Ls))
            L = Ls[l]

            xs = data["xs"][l]
            ts = (xs-xc)/xc # reduced temperatures
            ys = data[plot][l] # stuff to plot at this system size
            yerr = _err(plot, l)

            if plot == "binds":
                y_scaled = ys
                y_err_scaled = yerr
            elif plot == "chis":
                sf = L**(-gamma0 / nu0)
                y_scaled = ys * sf
                y_err_scaled = None if yerr is None else yerr * abs(sf)
            elif plot == "mags":
                sf = L**(beta0 / nu0)
                y_scaled = ys * sf
                y_err_scaled = None if yerr is None else yerr * abs(sf)
            elif _is_inv_time(plot):
                ys_inv = 1/ys
                y_scaled = ys_inv * L**(z0)
                # d(1/y)/dy = -1/y^2; then scaled by L^z.
                y_err_scaled = None if yerr is None else (L**(z0)) * yerr / np.where(ys == 0, np.nan, ys)**2

            eb = ax.errorbar(ts * Ls[l]**(1/nu0), y_scaled, yerr=y_err_scaled,
                             c=col, label=r'$%d$'%Ls[l], marker='o', mfc=col,
                             mew=mew, ms=ms, lw=lw*0, alpha=a,
                             elinewidth=0.8, capsize=2)
            lines0.append(eb.lines[0])
            ebars.append(eb)

        def update(nu,beta,gamma,z):
            print("nu       = ",nu)
            print("gamma    = ",gamma)
            print("beta     = ",beta)
            print("z        = ",z)
            print("beta(hs) = ",(d*nu-gamma)/2)
            print("eta (hs) = ",2-gamma/nu)
            print("Delta_ep = ",d-1/nu) # dimension of lightest Z2 neutral field
            print(" ")

            # will need to dynamically update the plot ranges 
            xmin = 1e10 
            xmax = -1e10
            ymin = 1e10 
            ymax = -1e10

            for l in range(len(Ls)):
                L = Ls[l]
                xdat = (data["xs"][l]-xc)/xc * L**(1/nu)

                yraw = data[plot][l]
                yerr = _err(plot, l)
                if plot == "chis":
                    sf = L**(-gamma/nu)
                    ydat = yraw * sf
                    ydat_err = None if yerr is None else yerr * abs(sf)
                elif plot == "mags":
                    sf = L**(beta/nu)
                    ydat = yraw * sf
                    ydat_err = None if yerr is None else yerr * abs(sf)
                elif plot == "binds":
                    ydat = yraw
                    ydat_err = yerr
                elif _is_inv_time(plot):
                    ydat = (1/yraw) * L**(z)
                    ydat_err = None if yerr is None else (L**z) * yerr / np.where(yraw == 0, np.nan, yraw)**2

                if ydat_err is not None:
                    fin = np.isfinite(ydat_err) & np.isfinite(ydat)
                    ymin = min(ymin, np.min(ydat - np.where(fin, ydat_err, 0)))
                    ymax = max(ymax, np.max(ydat + np.where(fin, ydat_err, 0)))
                else:
                    ymin = min(ymin, np.min(ydat))
                    ymax = max(ymax, np.max(ydat))
                xmin = min(xmin, np.min(xdat))
                xmax = max(xmax, np.max(xdat))

                # update the markers + error-bar segments
                lines0[l].set_xdata(xdat)
                lines0[l].set_ydata(ydat)
                _update_errorbar(ebars[l], xdat, ydat, ydat_err)
            
            ax.set_xlim(xmin*1.05,xmax*1.05)
            ax.set_ylim(ymin*.95,ymax*1.05)
            fig.canvas.draw_idle()
            
            # Dynamic title: "{rule name} | {exponent} = {value}" as a single
            # math-mode string. Strip the $...$ wrapper off the caller's
            # `title` so we can fold the rule name into one math segment
            # together with the exponent annotation.
            def _unc(name):
                # Render the uncertainty annotation that follows an exponent
                # value in the plot title:
                #   * symmetric (hi == lo, modulo float fuzz):  \pm <σ>
                #   * asymmetric                              :  ^{+hi}_{-lo}
                # Symmetric is the relevant case when σ_total has been
                # synthesised from bootstrap σ_stat and leave-one-L-out σ_sys,
                # which is the default since the plotter override that
                # collapses the asymmetric pair into ±σ_tot.
                if fit_uncert is None or name not in fit_uncert:
                    return ""
                lo, hi = fit_uncert[name]
                # treat differences below the rendering precision (~1%) as
                # symmetric; %.2g shows two sig figs anyway.
                tol = 0.01 * max(abs(lo), abs(hi), 1e-300)
                if abs(hi - lo) <= tol:
                    return r" \pm %.2g" % (0.5 * (hi + lo))
                return r"^{+%.2g}_{-%.2g}" % (hi, lo)

            rule_tex = title[1:-1] if title.startswith("$") and title.endswith("$") else title

            if plot == "binds":
                dynamic_title = r"$%s\, \, | \, \, \nu = %.3f%s$" % (rule_tex, nu, _unc("nu"))
            elif plot == "chis":
                dynamic_title = r"$%s\, \, | \, \, \gamma = %.3f%s$" % (rule_tex, gamma, _unc("gamma"))
            elif plot == "mags":
                dynamic_title = r"$%s\, \, | \, \, \beta = %.3f%s$" % (rule_tex, beta, _unc("beta"))
            elif _is_inv_time(plot):
                dynamic_title = r"$%s\, \, | \, \, z = %.3f%s$" % (rule_tex, z, _unc("z"))
            else:
                dynamic_title = title
            ax.set_title(dynamic_title)


        # define a function that updates values of critical exponents based on arrow key presses 
        def on_key(event):
            global nu,beta,gamma,z
            d = .005 # incriment by which exponents are tuned 
            numin = 0; numax = 4
            gammamin = 0; gammamax = 4
            betamin = 0; betamax = 4  
            zmin = .75; zmax = 4
            if event.key == 'l':
                z = max(zmin,z-d)
            elif event.key == ':':
                z = min(zmax,z+d)
            if event.key == 'left':
                nu = max(numin,nu - d) 
            elif event.key == 'right':
                nu = min(numax,nu + d) 
            elif event.key == 'up':
                gamma = min(gammamax,gamma+d)
            elif event.key == 'down':
                gamma = max(gammamin,gamma-d)
            elif event.key == '.': 
                beta = max(betamin,beta-d)
            elif event.key == ',': 
                beta = min(betamax,beta+d)

            update(nu,beta,gamma,z)
            
            
        fig.canvas.mpl_connect('key_press_event', lambda event : on_key(event))

        ax.legend(title=r'$L$',title_fontsize=14)
        if plot == "binds": 
            ylab = r'$B$'
        elif plot == "chis": 
            ylab = r'$L^{-\gamma/\nu}\chi$'
        elif plot == "mags": 
            ylab = r'$L^{\beta/\nu} m$'
        elif plot == "t_autos":
            ylab = r'$L^z / t_{\rm auto}$'
        elif plot == "t_rels":
            ylab = r'$L^z / t_{\rm rel}$'
        ax.set(xlabel=r'$\overline{p}L^{1/\nu}$',ylabel = ylab)
        # Seed the title + any other update()-driven state once, so the initial
        # figure isn't blank before the first keypress (matters now that the
        # title carries the exponent value).
        update(nu, beta, gamma, z)
    plt.show()
                


