import json
import numpy as np
import os


def read_series(folder):

    data = {}

    filename = os.path.join(folder,"series.jsonl")

    with open(filename,'r') as f:
        for line in f:
            row=json.loads(line)

            T=row["T"]
            M=row["M"]

            if T not in data:
                data[T]=[]

            data[T].append(M)

    return data



def calculate_chi(data,L):

    Ts=[]
    chis=[]


    for T in sorted(data.keys()):

        M=np.array(data[T])

        m_abs=np.abs(M)

        m2=M**2


        mean_abs=np.mean(m_abs)
        mean_m2=np.mean(m2)


        chi=L**2*(mean_m2-mean_abs**2)/T


        Ts.append(T)
        chis.append(chi)


    return np.array(Ts),np.array(chis)



def peak_fit(T,chi):


    # 最大susceptibility位置

    idx=np.argmax(chi)


    # 取附近5个点

    start=max(0,idx-2)
    end=min(len(T),idx+3)


    T_fit=T[start:end]
    chi_fit=chi[start:end]


    # 二次拟合

    coef=np.polyfit(T_fit,chi_fit,2)


    a,b,c=coef


    # 顶点

    T_peak=-b/(2*a)


    return T_peak



# ==========================
# 主程序
# ==========================


print("Reading data...")


data32=read_series(
"artifacts/window-l32"
)


data64=read_series(
"artifacts/window-l64"
)



T32,chi32=calculate_chi(data32,32)

T64,chi64=calculate_chi(data64,64)



Tpeak32=peak_fit(T32,chi32)

Tpeak64=peak_fit(T64,chi64)



Tc=2*Tpeak64-Tpeak32



print("====================")

print("Tpeak(L=32) =",Tpeak32)

print("Tpeak(L=64) =",Tpeak64)

print("Tc =",Tc)


# 保存结果

os.makedirs("evidence",exist_ok=True)


with open("evidence/peaks.txt","w") as f:

    f.write(
        f"Tpeak32 = {Tpeak32:.6f}\n"
    )

    f.write(
        f"Tpeak64 = {Tpeak64:.6f}\n"
    )

    f.write(
        f"Tc = {Tc:.6f}\n"
    )
