# Speed-of-light model for IR tensor-monomial canonicalization. All inputs explicit.
CPU = dict(P=8, f=5.0e9, iota=3.0, M=16, tmem=80e-9, B=70e9, br=17, pmis=0.5, L1=5)
GPU = dict(SM=24, f=2.46e9, int_per_clk_sm=64, B=272e9, pcie=13e9, warps=48, smem=100e3, shlat=30, eta=0.25)
cases = {  # m, n, d, f, g, k
 "A distinct (n=12)":      (3, 12, 6, 0, 3, 1),
 "B Riem^3 (n=12)":        (3, 12, 6, 0, 9, 3),
 "C quad-action (n=24)":   (4, 24, 11, 2, 6, 2),
 "D Riem^8 (n=32)":        (8, 32, 16, 0, 24, 8),
}
rows=[]
for name,(m,n,d,f,g,k) in cases.items():
    V=m+n+d+f+g; E=2*n+g
    Wb=4*n+2*E; Wr=8*E+6*V; Wl=3*(V+E); Wo=2*n
    Nind=(k-1)*(k+2)//2
    Ws= Nind*Wr/k + (k-1)*(Wl+2*(V+E)) if k>1 else 0.0
    Wsg=(k-1)*n
    W=Wb+Wr+Wl+Wo+Ws+Wsg
    splits = V + Nind*V/k
    cyc_thr = W/CPU["iota"]
    cyc_br  = CPU["pmis"]*CPU["br"]*splits
    rounds = 3*(1+Nind); cyc_lat = rounds*4*CPU["L1"]
    cyc = max(cyc_thr + cyc_br, cyc_lat)
    t1 = cyc/CPU["f"]
    tcpu = t1/CPU["P"]
    bin_ = 4*n+16; bout=bin_
    tbw = (bin_+bout+2*64)/CPU["B"]; tlat=CPU["tmem"]/(CPU["P"]*CPU["M"])
    tmem=max(tbw,tlat)
    tagg=max(tcpu,tmem)
    # GPU
    Rint=GPU["SM"]*GPU["int_per_clk_sm"]*GPU["f"]
    eta = GPU["eta"] if k<=3 else 0.1
    tg_comp=W/(eta*Rint)
    tg_dram=(bin_+bout+64)/GPU["B"]
    foot=80*n; inst=min(GPU["warps"], int(GPU["smem"]//foot))*GPU["SM"]
    tg_occ=rounds*4*GPU["shlat"]/GPU["f"]/inst
    tg_dev=max(tg_comp,tg_dram,tg_occ)
    tg_pcie=(bin_+bout)/GPU["pcie"]
    rows.append((name,V,E,Nind,W,cyc,t1,tcpu,tmem,tagg,tg_comp,tg_occ,tg_dev,tg_pcie,foot))
    print(f"{name:24s} V={V:3d} E={E:3d} Nind={Nind:3d} W={W:7.0f}uops cyc={cyc:7.0f} (thr {cyc_thr:.0f} br {cyc_br:.0f} lat {cyc_lat}) "
          f"t1={t1*1e9:6.1f}ns cpu8={tcpu*1e9:5.2f}ns mem={tmem*1e9:4.2f}ns agg={tagg*1e9:5.2f}ns | "
          f"GPUcomp={tg_comp*1e9:5.2f} occ={tg_occ*1e9:5.2f} dev={tg_dev*1e9:5.2f} pcie={tg_pcie*1e9:5.2f}ns foot={foot}B")

print("\n| Case | V | E | N_ind | W (µops) | ideal 1-core | practical 1-core | 8-core ideal /term | 8-core practical /term | memory floor /term |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name,(m,n,d,f,g,k) in cases.items():
    V=m+n+d+f+g; E=2*n+g
    Wb=4*n+2*E; Wr=8*E+6*V; Wl=3*(V+E); Wo=2*n
    Nind=(k-1)*(k+2)//2
    Ws= Nind*Wr/k + (k-1)*(Wl+2*(V+E)) if k>1 else 0.0
    W=Wb+Wr+Wl+Wo+Ws+(k-1)*n
    splits=V+Nind*V/k
    rounds=3*(1+Nind); clat=rounds*4*CPU["L1"]
    ci=max(W/CPU["iota"],clat); cp=max(W/CPU["iota"]+CPU["pmis"]*CPU["br"]*splits,clat)
    ti=ci/CPU["f"]; tp=cp/CPU["f"]
    b=4*n+16; tm=max((2*b+128)/CPU["B"], CPU["tmem"]/(CPU["P"]*CPU["M"]))
    print(f"| {name} | {V} | {E} | {Nind} | {W:,.0f} | {ti*1e9:.0f} ns | {tp*1e9:.0f} ns | {max(ti/8,tm)*1e9:.1f} ns | {max(tp/8,tm)*1e9:.1f} ns | {tm*1e9:.1f} ns |")
print("\n| Case | GPU compute (η) | GPU occupancy/latency | GPU DRAM | on-device floor /term | PCIe floor /term | offload floor /term | CPU 8-core practical /term |")
print("|---|---|---|---|---|---|---|---|")
for r,(name,(m,n,d,f,g,k)) in zip(rows,cases.items()):
    b=4*n+16; eta = GPU["eta"] if k<=3 else 0.1
    tdram=(2*b+64)/GPU["B"]
    print(f"| {name} | {r[10]*1e9:.2f} ns (η={eta}) | {r[11]*1e9:.2f} ns | {tdram*1e9:.2f} ns | {r[12]*1e9:.2f} ns | {r[13]*1e9:.1f} ns | {max(r[12],r[13])*1e9:.1f} ns | {r[9]*1e9:.1f} ns |")
