import csv 
import numpy as np 
import scipy.sparse as sp 
from numpy import ones as ons 
from numpy import zeros as zrs 
from numpy import insert as ist 
from numpy import cumsum as cmsm 
from numpy import arange as arng 

from .JPN_standard import jpn_standard

def opener(_box,_name): 
    with open(_name,'r') as _ : 
        _r = csv.reader(_) 
        for _row in _r : 
            _box.append(_row) 
def floater(_list): 
    for _i in range(len(_list)): 
        for _j in range(len(_list[_i])): 
            _list[_i][_j] = float(_list[_i][_j]) 
    return _list 
def transer(_label,_listIndex,_nely,_nelz): 
    _len = 1 
    _restore = [] 
    def R(_int): 
        return _restore.append(_label[_listIndex][_int]) 
    for _i in range(_len): # now only adapted for nely = 2,3,4,5,6 
        if _nely == 2. : 
            for _j in range(_nelz*1): R(_j*5+1) 
            for _j in range(_nelz*1): R(_j*5+2) 
            for _j in range(_nelz*1): R(_j*5+3) 
            for _j in range(_nelz*1): R(_j*5+4) 
            for _j in range(_nelz*2): R(_nelz*1*5+_j*3+1) 
            for _j in range(_nelz*2): R(_nelz*1*5+_j*3+2) 
        elif _nely == 3. : 
            for _j in range(_nelz*2): 
                if _j%2 == 0 : R(_j*5+1) 
                else : pass 
            for _j in range(_nelz*2): 
                if _j%2 == 0 : R(_j*5+2) 
                else : pass 
            for _j in range(_nelz*2): R(_j*5+3) 
            for _j in range(_nelz*2): R(_j*5+4) 
            for _j in range(_nelz*2): R(_nelz*2*5+_j*3+1) 
            for _j in range(_nelz*2): R(_nelz*2*5+_j*3+2) 
        elif _nely == 4. : 
            for _j in range(_nelz*2): 
                if _j%2 == 0 : R(_j*5+1) 
                else : pass 
            for _j in range(_nelz*2): 
                if _j%2 == 0 : R(_j*5+2) 
                else : pass 
            for _j in range(_nelz*2): R(_j*5+3) 
            for _j in range(_nelz*2): R(_j*5+4) 
            for _j in range(_nelz*3): R(_nelz*2*5+_j*3+1) 
            for _j in range(_nelz*3): R(_nelz*2*5+_j*3+2) 
        elif _nely == 5. : 
            for _j in range(_nelz*3): 
                if _j%3 == 0 : R(_j*5+1) 
                else : pass 
            for _j in range(_nelz*3): 
                if _j%3 == 0 : R(_j*5+2) 
                else : pass 
            for _j in range(_nelz*3): R(_j*5+3) 
            for _j in range(_nelz*3): R(_j*5+4) 
            for _j in range(_nelz*3): R(_nelz*3*5+_j*3+1)
            for _j in range(_nelz*3): R(_nelz*3*5+_j*3+2)
        elif _nely == 6. : 
            for _j in range(_nelz*3):
                if _j%3 == 0 : R(_j*5+1) 
                else : pass 
            for _j in range(_nelz*3): 
                if _j%3 == 0 : R(_j*5+2)
                else : pass
            for _j in range(_nelz*3): R(_j*5+3) 
            for _j in range(_nelz*3): R(_j*5+4) 
            for _j in range(_nelz*4): R(_nelz*3*5+_j*3+1) 
            for _j in range(_nelz*4): R(_nelz*3*5+_j*3+2) 
        else : pass 
    return _restore 
def modelParameter(_frame,_nelx,_nely,_nelz): 
    if _frame == 'x' : _nelxy = _nelx 
    elif _frame == 'y' : _nelxy = _nely 
    else : pass 
    _nj = (_nelxy+1)*(_nelz+1) ; _nc = (_nelxy+1)*_nelz 
    _ng = _nelxy*_nelz ; _nsj = _nelxy+1 
    return _nj,_nc,_ng,_nsj 
def rf_node_info(_frame,_nj,_nelx,_nely,_nelz): 
    _jd = [_i for _i in range(_nj)] 
    if _frame == 'x' : _nelxy = _nelx 
    elif _frame == 'y' : _nelxy = _nely 
    else : pass 
    del(_jd[0:(_nelxy+1)]) 
    del(_jd[0:len(_jd):_nelxy+1]) 
    _njr = arng(_nelxy+1,(_nelxy+1)*(_nelz+1),_nelxy+1).tolist()  
    _fnjd = _nelxy+1 
    _jf = zrs(shape=(_nj)) 
    for _i in range(_nj): 
        if _i in _jd : _jf[_i] = 3 
        else : _jf[_i] = 6         
    _njdp = [_i for _i in range(_nj)] 
    for _i in range(len(_njr)): 
        for _j in range(_fnjd): 
            _njdp[_njr[_i]+_j] = _njr[_i] 
    _njef = cmsm(_jf,dtype='int').tolist() 
    _njsf = [int(_njef[_i]) for _i in range(len(_njef)-1)] 
    _njsf = ist(_njsf,0,0).tolist()  
    _ndf = _njef[-1] 
    _njf = zrs(shape=(_nj*6)) 
    for _i in range(_nj): 
        if _jf[_i] == 6 : 
            for _j in range(6): 
                _njf[6*_i+_j] = arng(_njsf[_i],_njef[_i]+1)[_j] 
        else : 
            _njf[6*_i] = _njf[6*(_njdp[_i])] 
            _njf[6*_i+1] = _njf[6*(_njdp[_i])+1] 
            for _j in range(3): 
                _njf[6*_i+_j+2] = arng(_njsf[_i],_njef[_i]+1)[_j] 
            _njf[6*(_i+1)-1] = _njf[6*(_njdp[_i]+1)-1] 
    return _jd,_njr,_fnjd,_jf,_njdp,_njef,_njsf,_ndf,_njf 
def nodeCoordinate(_frame,_nj,_nelx,_nely,_nelz,_lx,_ly,_lz,_lz_first): 
    _x = zrs(shape=(_nj)) ; _y = _x.copy() ; _z = _x.copy() 
    _xe = [_lx for _i in range(_nelx)] ; _xe = ist(_xe,0,0).tolist() 
    _ye = [_ly for _i in range(_nely)] ; _ye = ist(_ye,0,0).tolist() 
    _ze = [_lz for _i in range(_nelz-1)] 
    _ze = ist(_ze,0,_lz_first) ; _ze = ist(_ze,0,0).tolist() 
    _xi = cmsm(_xe).tolist() ; _yi = cmsm(_ye).tolist() ; _zi = cmsm(_ze).tolist() 
    if _nelz == 0 : 
        print('\033[3m'+'\033[33m'+'\nError : Please input nelz > 0.'+'\033[0m') 
    else : 
        for _q in range(_nelz+1): 
            if _frame == 'x' : 
                _i = (_nelx+1)*_q+1 
                _j = (_nelx+1)*(_q+1) 
                for _w in range(_j-_i+1): 
                    _x[_i+_w-1] = _xi[_w] 
                    _z[_i+_w-1] = _zi[_q] 
            if _frame == 'y' : 
                _i = (_nely+1)*_q+1 
                _j = (_nely+1)*(_q+1) 
                for _w in range(_j-_i+1): 
                    _y[_i+_w-1] = _yi[_w] 
                    _z[_i+_w-1] = _zi[_q] 
    return _x,_y,_z,_xe,_ye,_ze,_xi,_yi,_zi 
def memberInfo(_frame,_nm,_nelx,_nely,_nelz): 
    _js = zrs(shape=(_nm)) ; _je = _js.copy() 
    _jel = _js.copy() ; _c_g = _js.copy() 
    if _frame == 'x' : _nelxy = _nelx 
    elif _frame == 'y' :_nelxy = _nely 
    else : pass 
    for _q in range(2,_nelz+2): 
        _i = _nelxy*(_q-2)+1  
        _j = _nelxy*(_q-1) 
        _s = (_nelxy+1)*(_q-1) 
        _t = (_nelxy+1)*_q 
        for _w in range(_j-_i+1): 
            _js[_i+_w-1] = arng(_s,_t-1)[_w] 
            _je[_i+_w-1] = arng(_s+1,_t)[_w] 
            _jel[_i+_w-1] = 1 ; _c_g[_i+_w-1] = 2 
    for _e in range(1,_nelxy+2): 
        _i = _nelxy*_nelz+_nelz*(_e-1)+1 
        _j = _nelxy*_nelz+_nelz*_e 
        _a = (_nelxy+1)*(_nelz-1)+_e 
        _b = _nelxy+1+_e 
        _c = (_nelxy+1)*_nelz+_e 
        for _r in range(_j-_i+1): 
            _js[_i+_r-1] = arng(_e-1,_a,_nelxy+1)[_r] 
            _je[_i+_r-1] = arng(_b-1,_c,_nelxy+1)[_r] 
            _jel[_i+_r-1] = 2 ; _c_g[_i+_r-1] = 1 
    return _js,_je,_jel,_c_g 
def memberLength(_nm,_x,_y,_z,_js,_je): 
    _lm = zrs(shape=(_nm)) 
    for _i in range(_nm): 
        _alx = _x[int(_je[_i])]-_x[int(_js[_i])] 
        _aly = _y[int(_je[_i])]-_y[int(_js[_i])] 
        _alz = _z[int(_je[_i])]-_z[int(_js[_i])] 
        _alj = np.sqrt(_alx**2+_aly**2+_alz**2) 
        _lm[_i] = _alj 
    return _lm 
def directionCosine(_nm,_js,_je,_x,_y,_z): 
    def _ystar(_xs,_ys,_zs,_xe,_ye,_ze,_angle): 
        _al = np.sqrt((_xe-_xs)**2+(_ye-_ys)**2+(_ze-_zs)**2) 
        _a1 = (_xe-_xs)/_al ; _a2 = (_ye-_ys)/_al ; _a3 = (_ze-_zs)/_al 
        _albar = np.sqrt(_a1**2+_a2**2) 
        if _albar != 0 : 
            _c1 = -_a2/_albar*np.cos(_angle)-_a3*_a1/_albar*np.sin(_angle) 
            _c2 = _a1/_albar*np.cos(_angle)-_a2*_a3/_albar*np.sin(_angle) 
            _c3 = _albar*np.sin(_angle) 
            _c = [_c1,_c2,_c3] 
        else : pass 
        if int(abs(_a3)) == 1 : 
            _c = [-np.sin(_angle),np.cos(_angle),0] 
        else : pass 
        return _c 
    _cyl = zrs(shape=(_nm,3)) 
    _ang = zrs(shape=(_nm)) 
    for _i in range(_nm): 
        _nns = int(_js[_i]) ; _nne = int(_je[_i]) ; _nan = np.pi*_ang[_i]/180 
        for _j in range(3): 
            _cyl[_i][_j] = _ystar(_x[_nns],_y[_nns],_z[_nns],
                                  _x[_nne],_y[_nne],_z[_nne],_nan)[_j] 
    return _cyl 
def bandwidth(_nm,_njsf,_njef,_js,_je): 
    _nb = 0 
    for _i in range(_nm): 
        _n = _njsf[int(_js[_i])]-_njef[int(_je[_i])] 
        if _n < 0 : _n = -_n 
        else : pass 
        if _n > _nb : _nb = _n 
        else : pass 
    return _nb 
def sectionGroup(_frame,_nelx,_nely,_nelz): 
    if _frame == 'x' : _nelxy = _nelx 
    elif _frame == 'y' : _nelxy = _nely 
    else : pass 
    _Hn = zrs(shape=(_nelxy*_nelz)) 
    _twn = _Hn.copy() 
    _Dn = zrs(shape=(_nelxy+1)*_nelz) 
    if _nelxy %2 == 0 : 
        for _i in range(_nelz): 
            for _j in range(_nelxy): 
                _Hn[_i*_nelxy+_j] = _i 
        _Bn = _Hn.copy() 
        for _i in range(_nelz): 
            for _j in range(int(_nelxy/2)): 
                _twn[_i*_nelxy+_j] = int(_nelxy/2)*_i+_j 
                _twn[_i*_nelxy+int(_nelxy/2)+_j] = int(_nelxy/2)*(_i+1)-_j-1 
        _tfn = _twn.copy() 
        for _i in range(int(_nelxy/2+1)*_nelz): 
            _Dn[_i] = _i 
        for _i in range(int(_nelxy/2)*_nelz): 
            _Dn[::-1][_i] = _i 
        for _i in range(int(_nelxy/2)): 
            _ = int(_nelxy/2+1)*_nelz 
            _Dn[_+_i*_nelz:_+(_i+1)*_nelz] = _Dn[_+_i*_nelz:_+(_i+1)*_nelz][::-1] 
        _tn = _Dn.copy() 
    elif _nelxy %2 == 1 : 
        for _i in range(_nelz): 
            for _j in range(_nelxy): 
                _Hn[_i*_nelxy+_j] = _i 
        _Bn = _Hn.copy() 
        for _i in range(_nelz): 
            for _j in range(int((_nelxy+1)/2)): 
                _twn[_i*_nelxy+_j] = int((_nelxy+1)/2)*_i+_j 
            for _j in range(int((_nelxy-1)/2)): 
                _twn[_i*_nelxy+int((_nelxy+1)/2)+_j] = int((_nelxy+1)/2)*_i+int((_nelxy-1)/2)-_j-1 
        _tfn = _twn.copy() 
        for _i in range(int((_nelxy+1)/2)*_nelz): 
            _Dn[_i] = _i 
        for _i in range(int((_nelxy+1)/2)*_nelz): 
            _Dn[::-1][_i] = _i 
        for _i in range(int((_nelxy+1)/2)): 
            _ = int((_nelxy+1)/2)*_nelz 
            _Dn[_+_i*_nelz:_+(_i+1)*_nelz] = _Dn[_+_i*_nelz:_+(_i+1)*_nelz][::-1] 
        _tn = _Dn.copy() 
    else : pass 
    return _Hn,_Bn,_twn,_tfn,_Dn,_tn 
def supportInfo(_nsj): 
    _ns = [_i for _i in range(_nsj)] 
    _isup = zrs(shape=(_nsj,6)) 
    _pd = _isup.copy() 
    return _ns,_isup,_pd 
def loadsf(_fht,_xi,_yi,_ndf,_c_g,_frame,_njf,_ng,_frame_io,_x,_y,_z,_je,_js,_cyl,
           _pw,_Z,_co,_lx,_ly,_nlc,_nelx,_nely,_nelz): 
    def earthquake(): 
        def _zrs(): return zrs(shape=(_nelz)) 
        _seismicForce = _zrs() 
        _Qi = _zrs() ; _Ci = _zrs() ; _Ai = _zrs() 
        _alpha = _zrs() ; _sigma = _zrs() 
        _T = 0.03*np.sum(_fht)*1e-3 
        _wi = _xi[-1]*_yi[-1]*_pw 
        if _T <= 0.6 : _Rt = 1. 
        elif 0.6 < _T <= 2*0.6 : _Rt = 1-0.2*(_T/0.6-1)**2 
        else : _Rt = 1.6*0.6/_T 
        for _i in range(_nelz): 
            if _i == 0 : _sigma[_i] = _wi 
            else : _sigma[_i] = _wi+_sigma[_i-1] 
        _sigma = _sigma[::-1] 
        for _i in range(_nelz): 
            _alpha[_i] = _sigma[_i]/_sigma[0] 
            _Ai[_i] = 1+(1/np.sqrt(_alpha[_i])-_alpha[_i])*(2*_T/(1+3*_T)) 
            _Ci[_i] = _Z*_Rt*_Ai[_i]*_co 
            _Qi[_i] = _Ci[_i]*_sigma[_i] 
        for _i in range(_nelz): 
            _j = _nelz-_i-1 
            if _i == 0 : _seismicForce[_i] = _Qi[_j] 
            else : _seismicForce[_i] = _Qi[_j]-_Qi[_j+1] 
        _seismicForce = _seismicForce[::-1] 
        return _seismicForce 
    def memberForce(): 
        _memberEnd = zrs(shape=(12,6)) 
        _M0 = zrs(shape=(6)) 
        if _lx < _ly : 
            _wxy = _pw*_lx/2 ; _Qx = _wxy*_lx/4 ; _Qy = _wxy*(_ly-_lx/2)/2 
            _Cx = -5*_wxy*_lx**2/96 ; _Cy = -_wxy*(_ly**3-2*((_lx/2)**2)*_ly+(_lx/2)**3)/(12*_ly) 
            _M0x = _wxy*_lx**2/12 ; _M0y = _wxy*(3*_ly**2-4*(_lx/2)**2)/24 
        elif _lx > _ly : 
            _wxy = _pw*_ly/2 ; _Qx = _wxy*(_lx-_ly/2)/2 ; _Qy = _wxy*_ly/4 
            _Cx = -_wxy*(_lx**3-2*((_ly/2)**2)*_lx+(_ly/2)**3)/(12*_lx) ; _Cy = -5*_wxy*_ly**2/96 
            _M0x = _wxy*(3*_lx**2-4*(_ly/2)**2)/24 ; _M0y = _wxy*_ly**2/12 
        else : 
            _wxy = _pw*_lx/2 ; _Qx = _wxy*_lx/4 ; _Qy = _wxy*_ly/4 
            _Cx = -5*_wxy*_lx**2/96 ; _Cy = -5*_wxy*_ly**2/96 
            _M0x = _wxy*_lx**2/12 ; _M0y = _wxy*_ly**2/12 
        _memberEnd[2][0] = _Qx ; _memberEnd[8][0] = _Qx 
        _memberEnd[4][0] = _Cx ; _memberEnd[10][0] = -_Cx 
        _memberEnd[:,1] = _memberEnd[:,0]*2 
        _memberEnd[2][2] = _Qy ; _memberEnd[8][2] = _Qy 
        _memberEnd[4][2] = _Cy ; _memberEnd[10][2] = -_Cy 
        _memberEnd[:,3] = _memberEnd[:,2]*2 
        _memberEnd[:,4] = _memberEnd[:,2]*4 
        _memberEnd[:,5] = _memberEnd[:,0]*4 
        _M0[0] = _M0x ; _M0[1] = _M0x*2 ; _M0[2] = _M0y 
        _M0[3] = _M0y*2 ; _M0[4] = _M0y*4 ; _M0[5] = _M0x*4 
        return _memberEnd,_M0 
    def arrangeLoad(_seismicForce,_memberEnd): 
        __f = zrs(shape=(_ndf,_nlc)) 
        _beam = [] 
        for _i in range(len(_c_g)): 
            if _c_g[_i] == 2 : _beam.append(_i) 
            else : pass 
        _fxj = zrs(shape=(_nelz,_nely+1)) 
        _fyj = zrs(shape=(_nelz,_nelx+1)) 
        _tt = zrs(shape=(3,3)) 
        if _frame == 'x' : 
            for _i in range(_nelz): 
                _fxj[_i][0] = (_nelx+1)*(_i+1) 
                __f[int(_njf[int(6*(_fyj[_i][0])+1)])][1] = _seismicForce[_i]/(_nely+1) 
            _lxyr = _fxj.T[0] 
            _nelxy = _nelx 
        elif _frame == 'y' : 
            for _i in range(_nelz): 
                _fyj[_i][0] = (_nely+1)*(_i+1) 
                __f[int(_njf[int(6*(_fyj[_i][0])+1)])][1] = _seismicForce[_i]/(_nelx+1) 
            _lxyr = _fyj.T[0] 
            _nelxy = _nely 
        for _i in range(_nlc): 
            for _j in range(_ng): 
                _lcase = _i ; _member = _beam[_j] 
                if _frame_io == 'i' : 
                    _mef = 3 ; _orthogout = 1 ; _orthogin = 5 
                elif _frame_io == 'o' : 
                    _mef = 2 ; _orthogout = 0 ; _orthogin = 1 
                else : pass 
                _alx = _x[int(_je[_member])]-_x[int(_js[_member])] 
                _aly = _y[int(_je[_member])]-_y[int(_js[_member])] 
                _alz = _z[int(_je[_member])]-_z[int(_js[_member])] 
                _al = np.sqrt(_alx**2+_aly**2+_alz**2) 
                _tt[0][0] = _alx/_al 
                _tt[1][0] = _aly/_al 
                _tt[2][0] = _alz/_al 
                _tt[0][1] = _cyl[_member][0] 
                _tt[1][1] = _cyl[_member][1] 
                _tt[2][1] = _cyl[_member][2] 
                _tt[0][2] = _tt[1][0]*_tt[2][1]-_tt[2][0]*_tt[1][1] 
                _tt[1][2] = _tt[2][0]*_tt[0][1]-_tt[0][0]*_tt[2][1] 
                _tt[2][2] = _tt[0][0]*_tt[1][1]-_tt[1][0]*_tt[0][1] 
                _nns = zrs(shape=(6)) ; _nne = zrs(shape=(6)) 
                for _k in range(6): 
                    _nns[_k] = _njf[int(6*(_js[_member])+_k)] 
                    _nne[_k] = _njf[int(6*(_je[_member])+_k)] 
                _listq = np.concatenate([np.dot(_tt,_memberEnd.T[_mef][0:3]),np.dot(_tt,_memberEnd.T[_mef][3:6])]) 
                _listw = np.concatenate([np.dot(_tt,_memberEnd.T[_mef][6:9]),np.dot(_tt,_memberEnd.T[_mef][9:12])]) 
                for _k in range(6): 
                    __f[int(_nne[_k])][_lcase] = __f[int(_nne[_k])][_lcase]+_listw[_k] 
                    __f[int(_nns[_k])][_lcase] = __f[int(_nns[_k])][_lcase]+_listq[_k] 
                if _j % _nelxy == 0 : 
                    for _k in range(3): 
                        __f[int(_nns[_k])][_lcase] = __f[int(_nns[_k])][_lcase]\
                                                    +np.dot(_tt,_memberEnd.T[_orthogout][0:3])[_k] 
                elif _j % _nelxy == _nelxy-1 : 
                    for _k in range(3): 
                        __f[int(_nne[_k])][_lcase] = __f[int(_nne[_k])][_lcase]\
                                                    +np.dot(_tt,_memberEnd.T[_orthogout][6:9])[_k] 
                else : 
                    for _k in range(3): 
                        __f[int(_nns[_k])][_lcase] = __f[int(_nns[_k])][_lcase]\
                                                    +np.dot(_tt,_memberEnd.T[_orthogin][0:3])[_k] 
                        __f[int(_nne[_k])][_lcase] = __f[int(_nne[_k])][_lcase]\
                                                    +np.dot(_tt,_memberEnd.T[_orthogin][6:9])[_k] 
        else : pass 
        return __f,_lxyr 
    seismicForce = earthquake() 
    memberEnd,M0 = memberForce() 
    f,lxyr = arrangeLoad(seismicForce,memberEnd) 
    return seismicForce,memberEnd,M0,f,lxyr 
def arrangeValue(_Hn,_Bn,_twn,_tfn,_Dn,_tn): 
    _Hp = [_i for _i in range(int(max(_Hn))+1)] 
    _Bp = [_i+max(_Hp)+1 for _i in range(int(max(_Bn))+1)] 
    _twp = [_i+max(_Bp)+1 for _i in range(int(max(_twn))+1)] 
    _tfp = [_i+max(_twp)+1 for _i in range(int(max(_tfn))+1)] 
    _Dp = [_i+max(_tfp)+1 for _i in range(int(max(_Dn))+1)] 
    _tp = [_i+max(_Dp)+1 for _i in range(int(max(_tn))+1)] 
    return _Hp,_Bp,_twp,_tfp,_Dp,_tp 
def sectionStandard(): 
    def flatten(_): 
        _flt = [] 
        for _i in _ : 
            if hasattr(_i,'__iter__') : _flt += flatten(_i) 
            else : _flt.append(_i) 
        return _flt 
    _jpnstd = jpn_standard
    # with open('JPN_standard_py.csv','r') as _ : 
    #     _reader = csv.reader(_) 
    #     for _row in _reader : _jpnstd.append(_row) 
    # for _i in range(len(_jpnstd)): _jpnstd[_i][0] = float(_jpnstd[_i][0]) 
    # _jpnstd = flatten(_jpnstd) 
    _He = _jpnstd[0:27] 
    _Be = _jpnstd[27:34] 
    _twe = _jpnstd[34:40] 
    _tfe = _jpnstd[40:49] 
    _De = _jpnstd[49:64] 
    _te = _jpnstd[64:74] 
    return _He,_Be,_twe,_tfe,_De,_te 
def varMember(_nvg,_nvc,_twn,_Dn): 
    _repg = zrs(shape=(1,_nvg)) ; _repc = zrs(shape=(1,_nvc)) 
    for _i in range(_nvg): 
        _repg[0][_i] = _twn.tolist().index(_i) 
    for _i in range(_nvc): 
        _repc[0][_i] = _Dn.tolist().index(_i) 
    return _repg,_repc 
def Constraints(_target,_nm,_ng,_nc,_frame,_frame_io,_c_g,_Hn,_Bn,_twn,_tfn,_Dn,_tn,_ndf,_nb,_je,_js,
                _x,_y,_z,_cyl,_aj,_E,_pr,_xr,_yr,_njf,_ns6,_nlc,_nsj,_ns,_isup,_pd,_memberEnd,_M0,_nelz,
                _nma,_F,_jel,_lm,_Hp,_Bp,_twp,_tfp,_Dp,_tp,_f,_nvg,_nvc,_repg,_repc,_lxyr,_fht,_nelx,_nely,_nj): 
    def datasf(__Ho,__Bo,__two,__tfo,__Do,__to): 
        _a = zrs(shape=(1,_nm)) 
        _asx = zrs(shape=(1,_ng)) ; _asy = _asx.copy() 
        _aiy = _a.copy() ; _aiz = _a.copy() 
        _zy = _a.copy() ; _zz = _a.copy() ; _zyf = _asx.copy() 
        _zpy = _a.copy() ; _zpz = _a.copy() 
        if _frame_io == 'i' : 
            for _i in range(_nm): 
                if _c_g[_i] == 1 : 
                    _D = __Do[int(_Dn[_i-_ng])] 
                    _t = __to[int(_tn[_i-_ng])] 
                    _a[0][_i] = _D**2-(_D-2*_t)**2 
                    _aiy[0][_i] = (_D**4-(_D-2*_t)**4)/12 
                    _aiz[0][_i] = _aiy[0][_i] 
                    _zy[0][_i] = _aiy[0][_i]/(_D/2) 
                    _zz[0][_i] = _zy[0][_i] 
                    _zpy[0][_i] = _D*_t*(_D-_t)+((_D-2*_t)**2)*_t/2 
                    _zpz[0][_i] = _zpy[0][_i] 
                else : 
                    _H = __Ho[int(_Hn[_i])] 
                    _B = __Bo[int(_Bn[_i])] 
                    _tw = __two[int(_twn[_i])] 
                    _tf = __tfo[int(_tfn[_i])] 
                    _a[0][_i] = _H*_B-(_B-_tw)*(_H-2*_tf) 
                    _asx[0][_i] = (_H-2*_tf)*_tw 
                    _asy[0][_i] = _B*_tf*2 
                    _aiy[0][_i] = (_B*_H**3-(_B-_tw)*(_H-2*_tf)**3)/12*1.5 
                    _aiz[0][_i] = (_tf*2*(_B**3)+(_H-2*_tf)*(_tw**3))/12 
                    _zy[0][_i] = _aiy[0][_i]/(_H/2) 
                    _zz[0][_i] = _aiz[0][_i]/(_B/2) 
                    _zyf[0][_i] = _B*(_H**3-(_H-2*_tf)**3)/(6*_H) 
                    _zpy[0][_i] = _B*_tf*(_H-_tf)+((_H-2*_tf)**2)*_tw/4 
                    _zpz[0][_i] = _tf*(_B**2)/2+(_H-2*_tf)*(_tw**2)/4 
        else : 
            for _i in range(_nm): 
                if _c_g[0][_i] == 1 : 
                    _D = __Do[int(_Dn[_i-_ng])] 
                    _t = __to[int(_tn[_i-_ng])] 
                    _a[0][_i] = _D**2-(_D-2*_t)**2 
                    _aiy[0][_i] = (_D**4-(_D-2*_t)**4)/12 
                    _aiz[0][_i] = _aiy[0][_i] 
                    _zy[0][_i] = _aiy[0][_i]/(_D/2) 
                    _zz[0][_i] = _zy[0][_i] 
                    _zpy[0][_i] = _D*_t*(_D-_t)+((_D-2*_t)**2)*_t/2 
                    _zpz[0][_i] = _zpy[0][_i] 
                else : 
                    _H = __Ho[int(_Hn[_i])] 
                    _B = __Bo[int(_Bn[_i])] 
                    _tw = __two[int(_twn[_i])] 
                    _tf = __tfo[int(_tfn[_i])] 
                    _a[0][_i] = _H*_B-(_B-_tw)*(_H-2*_tf) 
                    _asx[0][_i] = (_H-2*_tf)*_tw 
                    _asy[0][_i] = _B*_tf*2 
                    _aiy[0][_i] = (_B*_H**3-(_B-_tw)*(_H-2*_tf)**3)/12*1.3 
                    _aiz[0][_i] = (_tf*2*(_B**3)+(_H-2*_tf)*(_tw**3))/12 
                    _zy[0][_i] = _aiy[0][_i]/(_H/2) 
                    _zz[0][_i] = _aiz[0][_i]/(_B/2) 
                    _zyf[0][_i] = _B*(_H**3-(_H-2*_tf)**3)/(6*_H) 
                    _zpy[0][_i] = _B*_tf*(_H-_tf)+((_H-2*_tf)**2)*_tw/4 
                    _zpz[0][_i] = _tf*(_B**2)/2+(_H-2*_tf)*(_tw**2)/4 
        return _a,_asx,_asy,_aiy,_aiz,_zy,_zz,_zyf,_zpy,_zpz
    def stifsf(_a,_aiy,_aiz): 
        def estfsf(__alj,__aaj,__aiyj,__aizj,__ajj,E,pr): 
            _es = zrs(shape=(12,12)) 
            _g = E/(2*(1+pr)) 
            _es[0][0] = E*__aaj/__alj ; _es[6][0] = -_es[0][0] 
            _es[6][6] = _es[0][0] ; _es[1][1] = 12*E*__aizj/(__alj**3) 
            _es[7][1] = -_es[1][1] ; _es[7][7] = _es[1][1] 
            _es[2][2] = 12*E*__aiyj/(__alj**3) ; _es[8][2] = -_es[2][2] 
            _es[8][8] = _es[2][2] ; _es[3][3] = _g*__ajj/__alj 
            _es[9][3] = -_es[3][3] ; _es[9][9] = _es[3][3] 
            _es[5][1] = 6*E*__aizj/(__alj**2) ; _es[11][1] = _es[5][1] 
            _es[11][7] = -_es[5][1] ; _es[4][2] = -6*E*__aiyj/(__alj**2) 
            _es[10][2] = _es[4][2] ; _es[10][8] = -_es[10][2] 
            _es[8][4] = 6*E*__aiyj/(__alj**2) ; _es[10][4] = 2*E*__aiyj/__alj 
            _es[7][5] = -6*E*__aizj/(__alj**2) ; _es[11][5] = 2*E*__aizj/__alj 
            _es[4][4] = 4*E*__aiyj/__alj ; _es[10][10] = _es[4][4] 
            _es[5][5] = 4*E*__aizj/__alj ; _es[11][11] = _es[5][5] 
            for _i in range(12): 
                for _j in range(12): 
                    _es[_i][_j] = _es.T[_i][_j] 
            return _es 
        _s = zrs(shape=(_ndf,_nb)) ; _t = zrs(shape=(3,3)) 
        for _i in range(_nm): 
            _alx = _x[int(_je[_i])]-_x[int(_js[_i])] 
            _aly = _y[int(_je[_i])]-_y[int(_js[_i])] 
            _alz = _z[int(_je[_i])]-_z[int(_js[_i])] 
            _alj = np.sqrt(_alx**2+_aly**2+_alz**2) 
            _t[0][0] = _alx/_alj 
            _t[0][1] = _aly/_alj 
            _t[0][2] = _alz/_alj 
            _t[1][0] = _cyl[_i][0] 
            _t[1][1] = _cyl[_i][1] 
            _t[1][2] = _cyl[_i][2] 
            _t[2][0] = _t[0][1]*_t[1][2]-_t[0][2]*_t[1][1] 
            _t[2][1] = _t[0][2]*_t[1][0]-_t[0][0]*_t[1][2] 
            _t[2][2] = _t[0][0]*_t[1][1]-_t[0][1]*_t[1][0] 
            _tt = _t.T 
            _aaj = _a[0][_i] ; _aiyj = _aiy[0][_i] 
            _aizj = _aiz[0][_i] ; _ajj = _aj[_i][0] 
            _es = estfsf(_alj,_aaj,_aiyj,_aizj,_ajj,_E,_pr) 
            
            _h = zrs(shape=(3,3)) 
            for _q in range(4): 
                for _w in range(4): 
                    for _e in range(3): 
                        for _r in range(3): 
                            _ies = 3*_q+_e ; _jes = 3*_w+_r 
                            _h[_e][_r] = _es[_ies,_jes] 
                    _h = np.dot(_tt,_h) ; _h = np.dot(_h,_t) 
                    for _e in range(3): 
                        for _r in range(3): 
                            _ies = 3*_q+_e ; _jes = 3*_w+_r 
                            _es[_ies,_jes] = _h[_e][_r] 
            
            for _q in range(6): 
                _es.T[5][_q] = _es.T[5][_q]-_es.T[0][_q]*_yr[int(_js[_i])]\
                                           +_es.T[1][_q]*_xr[int(_js[_i])] 
            for _q in range(12): 
                _es.T[11][_q] = _es.T[11][_q]-_es.T[6][_q]*_yr[int(_je[_i])]\
                                             +_es.T[7][_q]*_xr[int(_je[_i])] 
            _es[5][5] = _es[5][5]-_es[0][5]*_yr[int(_js[_i])]+_es[1][5]*_xr[int(_js[_i])] 
            _es[5][11] = _es[5][11]-_es[0][11]*_yr[int(_js[_i])]+_es[1][11]*_xr[int(_js[_i])] 
            _es[11][11] = _es[11][11]-_es[6][11]*_yr[int(_je[_i])]+_es[7][11]*_xr[int(_je[_i])] 
            _es[5][6] = _es[5][6]-_es[0][6]*_yr[int(_js[_i])]+_es[1][6]*_xr[int(_js[_i])] 
            _es[5][7] = _es[5][7]-_es[0][7]*_yr[int(_js[_i])]+_es[1][7]*_xr[int(_js[_i])] 
            _es[5][8] = _es[5][8]-_es[0][8]*_yr[int(_js[_i])]+_es[1][8]*_xr[int(_js[_i])] 
            _es[5][9] = _es[5][9]-_es[0][9]*_yr[int(_js[_i])]+_es[1][9]*_xr[int(_js[_i])] 
            _es[5][10] = _es[5][10]-_es[0][10]*_yr[int(_js[_i])]+_es[1][10]*_xr[int(_js[_i])] 
            for _q in range(5): _es[5][_q] = _es.T[5][_q] 
            for _q in range(11): _es[11][_q] = _es.T[11][_q] 
            for _q in range(2): _es.T[5][6+_q] = _es[5][6+_q] 
            
            _im = zrs(shape=(1,12)) 
            for _q in range(6): 
                _im[0][_q] = _njf[int(6*_js[_i])+_q] 
                _im[0][6+_q] = _njf[int(6*_je[_i])+_q] 
            for _q in range(12): 
                for _w in range(12): 
                    _k = int(_im[0][_w]-_im[0][_q]) 
                    if _k >= 0 : 
                        _s[int(_im[0][_q])][_k] += _es[_q][_w] 
                    else : pass 
        return _s 
    def suptsf(__s,__f): 
        _sk = zrs(shape=(1,_ns6)) ; _r = zrs(shape=(_ns6,_nlc)) 
        for _i in range(_nsj): 
            _m1 = 6*_i ; _n1 = 6*_ns[_i] 
            for _j in range(6): 
                if _isup[_i][_j] == 0 : 
                    _m = _m1+_j ; _n = _n1+_j 
                    _sk[0][_m] = __s[_n][0] 
                    __s[_n][0] = __s[_n][0]*1e4 
                    for _k in range(_nlc): 
                        _r[_m][_k] = __f[_n][_k] 
                        __f[_n][_k] = -_sk[0][_m]*1e4*_pd[_i][_j] 
        return __s,__f,_r,_sk 
    def eqsoln(__s,__f): 
        for _i in range(1,_nb): 
            __s.T[_i][_i:_ndf] = __s.T[_i][:_ndf-_i] 
        _ss = sp.spdiags(__s.T,list(range(0,_nb)),_ndf,_ndf) 
        _sd = sp.tril(_ss.T,k=-1) 
        _ss = (_ss+_sd).toarray() 
        _L = np.linalg.cholesky(_ss) 
        _t = np.linalg.solve(_L,(-__f)) 
        _d = np.linalg.solve(_L.T.conj(),_t) 
        return _d 
    def rsltsf(__r,__sk,__d): 
        def estfsf(__alj,__aaj,__aiyj,__aizj,__ajj,E,pr): 
            _es = zrs(shape=(12,12)) 
            _g = E/(2*(1+pr)) 
            _es[0][0] = E*__aaj/__alj ; _es[6][0] = -_es[0][0] 
            _es[6][6] = _es[0][0] ; _es[1][1] = 12*E*__aizj/(__alj**3) 
            _es[7][1] = -_es[1][1] ; _es[7][7] = _es[1][1] 
            _es[2][2] = 12*E*__aiyj/(__alj**3) ; _es[8][2] = -_es[2][2] 
            _es[8][8] = _es[2][2] ; _es[3][3] = _g*__ajj/__alj 
            _es[9][3] = -_es[3][3] ; _es[9][9] = _es[3][3] 
            _es[5][1] = 6*E*__aizj/(__alj**2) ; _es[11][1] = _es[5][1] 
            _es[11][7] = -_es[5][1] ; _es[4][2] = -6*E*__aiyj/(__alj**2) 
            _es[10][2] = _es[4][2] ; _es[10][8] = -_es[10][2] 
            _es[8][4] = 6*E*__aiyj/(__alj**2) ; _es[10][4] = 2*E*__aiyj/__alj 
            _es[7][5] = -6*E*__aizj/(__alj**2) ; _es[11][5] = 2*E*__aizj/__alj 
            _es[4][4] = 4*E*__aiyj/__alj ; _es[10][10] = _es[4][4] 
            _es[5][5] = 4*E*__aizj/__alj ; _es[11][11] = _es[5][5] 
            for _i in range(12): 
                for _j in range(12): 
                    _es[_i][_j] = _es.T[_i][_j] 
            return _es 
        _RS = zrs(shape=(12*_nlc,_nm)) 
        __Mc = zrs(shape=(_nlc,_nm)) 
        for _i in range(_nlc): 
            for _j in range(_nsj): 
                _m1 = 6*_j ; _n1 = 6*_ns[_j] 
                for _k in range(6): 
                    if _isup[_j][_k] == 0 : 
                        _m = _m1+_k ; _n = _n1+_k 
                        __r[_m][_i] += __sk[0][_m]*1e4*(_pd[_j][_k]-__d[_n][_i])\
                                      +__sk[0][_m]*__d[_n][_i] 
            _t = zrs(shape=(3,3)) 
            for _j in range(_nm): 
                _alx = _x[int(_je[_j])]-_x[int(_js[_j])] 
                _aly = _y[int(_je[_j])]-_y[int(_js[_j])] 
                _alz = _z[int(_je[_j])]-_z[int(_js[_j])] 
                _alj = np.sqrt(_alx**2+_aly**2+_alz**2) 
                _t[0][0] = _alx/_alj 
                _t[0][1] = _aly/_alj 
                _t[0][2] = _alz/_alj 
                _t[1][0] = _cyl[_j][0] 
                _t[1][1] = _cyl[_j][1] 
                _t[1][2] = _cyl[_j][2] 
                _t[2][0] = _t[0][1]*_t[1][2]-_t[0][2]*_t[1][1] 
                _t[2][1] = _t[0][2]*_t[1][0]-_t[0][0]*_t[1][2] 
                _t[2][2] = _t[0][0]*_t[1][1]-_t[0][1]*_t[1][0] 
                _aaj = _a[0][_j] ; _aiyj = _aiy[0][_j] 
                _aizj = _aiz[0][_j] ; _ajj = _aj[_j][0] 
                _es = estfsf(_alj,_aaj,_aiyj,_aizj,_ajj,_E,_pr) 
                _nd = zrs(shape=(1,12)) 
                for _k in range(6): 
                    _nd[0][_k] = _njf[int(6*_js[_j])+_k] 
                    _nd[0][6+_k] = _njf[int(6*_je[_j])+_k] 
                _dt = zrs(shape=(12,1)) 
                _h = [] 
                for _k in range(12): 
                    _h.append(__d[int(_nd[0][_k])][_i]) 
                _h[0] -= _yr[int(_js[_j])]*_h[5] 
                _h[1] += _xr[int(_js[_j])]*_h[5] 
                _h[6] -= _yr[int(_je[_j])]*_h[11] 
                _h[7] += _xr[int(_je[_j])]*_h[11] 
                for _k in range(4): 
                    _ii = [_i+3*_k for _i in range(3)] 
                    _hh = _h[_ii[0]:_ii[-1]+1] 
                    for _l in range(3): 
                        _dt[_ii[0]+_l][0] = np.dot(_t,np.array(_hh).reshape(3,1))[_l] 
                _arm = np.dot(_es,_dt) 
                if _j <= _ng : 
                    if _frame == 'x' : 
                        if _frame_io == 'o' : _mef = 0 
                        elif _frame_io == 'i' : _mef = 1 
                        else : pass 
                        for _k in range(12): 
                            _arm.T[0][_k] += _memberEnd.T[_mef][_k] 
                        __Mc[_i][_j] = _M0[_mef] 
                    elif _frame == 'y' : 
                        if _frame_io == 'o' : _mef = 2 
                        elif _frame_io == 'i' : _mef = 3 
                        else : pass 
                        for _k in range(12): 
                            _arm.T[0][_k] += _memberEnd.T[_mef][_k] 
                        __Mc[_i][_j] = _M0[_mef] 
                    else : pass 
                else : pass 
                for _k in range(12): 
                    _RS[12*_i+_k][_j] = _arm[_k][0] 
                __Mc[_i][_j] += (_arm[4][0]-_arm[10][0])/2 
            _Mc = zrs(shape=(_nlc,_ng)) 
            for _j in range(_nlc): 
                _Mc[_j][:] = __Mc[_j][:_ng] 
        return _arm,_RS,_Mc 
    def stress(__RS,__Mc,_a,_asx,_asy,_zy,_zz,_zyf): 
        _st = zrs(shape=(12*_nlc,_nm)) 
        _stc = zrs(shape=(_nlc,_ng)) 
        for _i in range(_nm): 
            for _j in range(_nlc): 
                _st[12*_j][_i] = __RS[12*_j][_i]/_a[0][_i] 
                _st[12*_j+5][_i] = __RS[12*_j+5][_i]/_zz[0][_i] 
                _st[12*_j+6][_i] = __RS[12*_j+6][_i]/_a[0][_i] 
                _st[12*_j+11][_i] = __RS[12*_j+11][_i]/_zz[0][_i] 
                if _i < _ng : 
                    _st[12*_j+1][_i] = __RS[12*_j+1][_i]/_asy[0][_i] 
                    _st[12*_j+2][_i] = __RS[12*_j+2][_i]/_asx[0][_i] 
                    _st[12*_j+4][_i] = __RS[12*_j+4][_i]/_zyf[0][_i] 
                    _st[12*_j+7][_i] = __RS[12*_j+7][_i]/_asy[0][_i] 
                    _st[12*_j+8][_i] = __RS[12*_j+8][_i]/_asx[0][_i] 
                    _st[12*_j+10][_i] = __RS[12*_j+10][_i]/_zyf[0][_i] 
                    _stc[_j][_i] = _Mc[_j][_i]/_zyf[0][_i] 
                else : 
                    _st[12*_j+1][_i] = __RS[12*_j+1][_i]/_a[0][_i] 
                    _st[12*_j+2][_i] = __RS[12*_j+2][_i]/_a[0][_i] 
                    _st[12*_j+4][_i] = __RS[12*_j+4][_i]/_zy[0][_i] 
                    _st[12*_j+7][_i] = __RS[12*_j+7][_i]/_a[0][_i] 
                    _st[12*_j+8][_i] = __RS[12*_j+8][_i]/_a[0][_i] 
                    _st[12*_j+10][_i] = __RS[12*_j+10][_i]/_zy[0][_i] 
        return _st,_stc 
    def gwidth(__Ho,__Bo): 
        _gb = zrs(shape=(1,_nelz-1)) ; _gw = _gb.copy() 
        for _i in range(_nelz-1): 
            _gb[0][_i] = __Ho[_i+1]/__Ho[_i]-1 
            _gw[0][_i] = __Bo[_i+1]/__Bo[_i]-1 
        return _gb,_gw 
    def stress_ratio(__Ho,__Bo,__two,__tfo,__st,__stc,_a,_aiy,_aiz): 
        _ratio = zrs(shape=(12*_nlc,_nm)) 
        _ratioc = zrs(shape=(_nlc,_ng)) 
        _ft = zrs(shape=(2,_nm)) ; _fs = _ft.copy() 
        _fc = zrs(shape=(2,_nm)) ; _fb = _fc.copy() 
        _LAM = zrs(shape=(1,_nma)) 
        for _i in range(_nma): 
            _LAM[0][_i] = np.pi*(_E/(0.6*_F[_i]))**0.5 
        for _i in range(_nm): 
            if _jel[_i] != 1 : _idx = 1 
            else : _idx = 0 
            _ft[0][_i] = _F[_idx]/1.5 ; _ft[1][_i] = _F[_idx] 
            _fs[0][_i] = _F[_idx]/(1.5*np.sqrt(3)) ; _fs[1][_i] = _fs[0][_i]*1.5 
        _iy = zrs(shape=(1,_nm)) ; _iz = _iy.copy() 
        _lambday = _iy.copy() ; _lambdaz = _iy.copy() 
        _xn = ons(shape=(1,_nc)) ; _fx = _xn.copy() 
        _an = zrs(shape=(1,_nc)) ; _bn = _an.copy() 
        for _i in range(_nc): 
            _j = _i+_ng 
            _memA = [] ; _memB = [] 
            for _k in range(_nm): 
                if _js[_k] == _js[_j] or _je[_k] == _js[_j] : _memA.append(_k) 
                else : pass 
                if _js[_k] == _je[_j] or _je[_k] == _je[_j] : _memB.append(_k) 
                else : pass 
            del _memA[-1] ; del _memB[-1] 
            _gn = _aiy[0][_j]/_lm[_j] 
            if _memA == [] : _Ga = 1.0 
            else : 
                _gca = _aiy[0][max(_memA)]/_lm[max(_memA)] 
                if len(_memA) == 3 : 
                    _gba1 = _aiy[0][_memA[0]]/_lm[_memA[0]] 
                    _gba2 = _aiy[0][_memA[1]]/_lm[_memA[1]] 
                else : 
                    _gba1 = _aiy[0][_memA[0]]/_lm[_memA[0]] 
                    _gba2 = 0 
                _Ga = (_gn+_gca)/(_gba1+_gba2) 
            if max(_memB) < _ng : 
                if len(_memB) == 1 : 
                    _gbb1 = _aiy[0][_memB[0]]/_lm[_memB[0]] 
                    _gbb2 = 0 ; _gcb = 0 
                else : 
                    _gbb1 = _aiy[0][_memB[0]]/_lm[_memB[0]] 
                    _gbb2 = _aiy[0][_memB[1]]/_lm[_memB[1]] 
                    _gcb = 0 
            elif len(_memB) == 3 : 
                _gbb1 = _aiy[0][_memB[0]+1]/_lm[_memB[0]+1] 
                _gbb2 = _aiy[0][_memB[1]+1]/_lm[_memB[1]+1] 
                _gcb = _aiy[0][_memB[2]+1]/_lm[_memB[2]+1] 
            else : 
                _gbb1 = _aiy[0][_memB[0]]/_lm[_memB[0]] 
                _gbb2 = 0 
                _gcb = _aiy[0][_memB[1]]/_lm[_memB[1]] 
            _Gb = (_gn+_gcb)/(_gbb1+_gbb2) 
            _an[0][_i] = -1*_Ga*_Gb/(6*(_Ga+_Gb)) 
            _bn[0][_i] = -1*6/(_Ga+_Gb) 
        while True : 
            _fx = _xn/np.tan(_xn)+_an*(_xn**2)-_bn 
            _dfx = 1/np.tan(_xn)-_xn/((np.sin(_xn))**2)+2*_an*_xn 
            _xn = _xn -_fx/_dfx 
            if max(abs(_fx[0])) < 1e-3 : 
                break 
        _kc = np.pi/_xn 
        _lk = _lm.copy() ; _lk[_ng:_ng+_nc+1] = _kc[0]*_lm[_ng:_ng+_nc+1] 
        for _i in range(_nm): 
            _iy[0][_i] = (_aiy[0][_i]/_a[0][_i])**0.5 
            _iz[0][_i] = (_aiz[0][_i]/_a[0][_i])**0.5 
            _lambday[0][_i] = _lk[_i]/_iy[0][_i] 
            _lambdaz[0][_i] = _lk[_i]/_iz[0][_i] 
            _lambda = max(_lambday[0][_i],_lambdaz[0][_i]) 
            if _lambda <= _LAM[0][int(_jel[_i])-1] : 
                _nu = 1.5+2/3*(_lambda/_LAM[0][int(_jel[_i])-1])**2 
                _fc[0][_i] = _F[int(_jel[_i])-1]/_nu*(1-0.4*(_lambdaz[0][_i]/_LAM[0][int(_jel[_i])-1])**2) 
            else : 
                _fc[0][_i] = 0.277*_F[int(_jel[_i])-1]/(_lambda/_LAM[0][int(_jel[_i])-1])**2 
        _Lb = zrs(shape=(1,_nm)) ; _nstif = _Lb.copy() 
        for _i in range(_nm): 
            if _c_g[_i] == 1 : 
                _nstif[0][_i] = 0 ; _Lb[0][_i] = _lm[_i] 
            else : 
                if _F[int(_jel[_i])] == 235 : 
                    _nstif[0][_i] = np.ceil(max((_lambdaz[0][_i]-170)/20,_lm[_i]/3000-1)) 
                    _Lb[0][_i] = _lm[_i]/(_nstif[0][_i]+1) 
                elif _F[int(_jel[_i])] == 325 : 
                    _nstif[0][_i] = np.ceil(max((_lambdaz[0][_i]-130)/20,_lm[_i]/3000-1)) 
                    _Lb[0][_i] = _lm[_i]/(_nstif[0][_i]+1) 
                else : _nstif[0][_i] = 0 
        for _i in range(_nm): 
            if _c_g[_i] == 2 : 
                _H = __Ho[int(_Hn[_i])] 
                _B = __Bo[int(_Bn[_i])] 
                _tw = __two[int(_twn[_i])] 
                _tf = __tfo[int(_tfn[_i])] 
                _lbi = _Lb[0][_i] 
                _siy = np.sqrt((_tf*_B**3/12)/(_tf*_B+(_H/6-_tf)*_tw)) 
                _c = 1.0 
                _fb1 = (1-0.4*(_lbi/_siy)**2/(_c*_LAM[0][int(_jel[_i])]**2))*_ft[0][_i] 
                _fb2 = 89000/(_lbi*_H/(2*_tf*_B)) 
                _fb[0][_i] = max(_fb1,_fb2) 
                if _fb[0][_i] > _ft[0][_i] : 
                    _fb[0][_i] = _ft[0][_i] 
                else : pass 
            else : _fb[0][_i] = _ft[0][_i] 
        for _i in range(_nm): 
            _fb[1][_i] = _fb[0][_i]*1.5 
            _fc[1][_i] = _fc[0][_i]*1.5 
        for _i in range(_nm): 
            if __st[0][_i] >= 0 : 
                _ratio[0][_i] = _st[0][_i]/_ft[0][_i] 
                _ratio[6][_i] = _st[6][_i]/_ft[0][_i] 
            else : 
                _ratio[0][_i] = _st[0][_i]/_fc[0][_i] 
                _ratio[6][_i] = _st[6][_i]/_fc[0][_i] 
            _ratio[1][_i] = _st[1][_i]/_fs[0][_i] 
            _ratio[2][_i] = _st[2][_i]/_fs[0][_i] 
            _ratio[4][_i] = _st[4][_i]/_fb[0][_i] 
            _ratio[5][_i] = _st[5][_i]/_ft[0][_i] 
            _ratio[7][_i] = _st[7][_i]/_fs[0][_i] 
            _ratio[8][_i] = _st[8][_i]/_fs[0][_i] 
            _ratio[10][_i] = _st[10][_i]/_fb[0][_i] 
            _ratio[11][_i] = _st[11][_i]/_ft[0][_i] 
            if _i < _ng : 
                _ratioc[0][_i] = __stc[0][_i]/_fb[0][_i] 
            else : pass 
            if _nlc > 1 : 
                for _j in range(_nlc-1): 
                    if __st[12*(_j+1)][_i] >= 0 : 
                        _ratio[12*(_j+1)][_i] = __st[12*(_j+1)][_i]/_ft[1][_i] 
                        _ratio[12*(_j+1)+6][_i] = __st[12*(_j+1)+6][_i]/_ft[1][_i] 
                    else : 
                        _ratio[12*(_j+1)][_i] = __st[12*(_j+1)][_i]/_fc[1][_i] 
                        _ratio[12*(_j+1)+6][_i] = __st[12*(_j+1)+6][_i]/_fc[1][_i] 
                    _ratio[12*(_j+1)+1][_i] = __st[12*(_j+1)+1][_i]/_fs[1][_i] 
                    _ratio[12*(_j+1)+2][_i] = __st[12*(_j+1)+2][_i]/_fs[1][_i] 
                    _ratio[12*(_j+1)+4][_i] = __st[12*(_j+1)+4][_i]/_fb[1][_i] 
                    _ratio[12*(_j+1)+5][_i] = __st[12*(_j+1)+5][_i]/_ft[1][_i] 
                    _ratio[12*(_j+1)+7][_i] = __st[12*(_j+1)+7][_i]/_fs[1][_i] 
                    _ratio[12*(_j+1)+8][_i] = __st[12*(_j+1)+8][_i]/_fs[1][_i] 
                    _ratio[12*(_j+1)+10][_i] = __st[12*(_j+1)+10][_i]/_fb[1][_i] 
                    _ratio[12*(_j+1)+11][_i] = __st[12*(_j+1)+11][_i]/_ft[1][_i] 
                    if _i < _ng : 
                        _ratioc[_j+1][_i] = __stc[_j+1][_i]/_fb[1][_i] 
                    else : pass 
        _bri = zrs(shape=(1,_ng*_nlc)) ; _brj = _bri.copy() ; _brc = _bri.copy() 
        _bsi = _bri.copy() ; _bsj = _bri.copy() 
        _cr = zrs(shape=(1,_nc*_nlc)) ; _cs = _cr.copy() 
        for _i in range(_nlc): 
            for _j in range(_nm): 
                if _c_g[_j] == 2 : 
                    _idx = _ng*_i+_j 
                    _bi1 = abs(_ratio[12*_i+4][_j]) 
                    _bi2 = abs(_ratio[12*_i+5][_j]) 
                    _bj1 = abs(_ratio[12*_i+10][_j]) 
                    _bj2 = abs(_ratio[12*_i+11][_j]) 
                    _bri[0][_idx] = max(_bi1,_bi2)-1 
                    _brj[0][_idx] = max(_bj1,_bj2)-1 
                    _brc[0][_idx] = abs(_ratioc[_i][_j])-1 
                    _bsi1 = abs(_ratio[12*_i+1][_j]) 
                    _bsi2 = abs(_ratio[12*_i+2][_j]) 
                    _bsj1 = abs(_ratio[12*_i+7][_j]) 
                    _bsj2 = abs(_ratio[12*_i+8][_j]) 
                    _bsi[0][_idx] = max(_bsi1,_bsi2)-1 
                    _bsj[0][_idx] = max(_bsj1,_bsj2)-1 
                else : 
                    _jdx = _nc*_i+_j-_ng 
                    _cc = abs(_ratio[12*_i][_j]) 
                    _cbi1 = abs(_ratio[12*_i+4][_j]) 
                    _cbi2 = abs(_ratio[12*_i+5][_j]) 
                    _cbj1 = abs(_ratio[12*_i+10][_j]) 
                    _cbj2 = abs(_ratio[12*_i+11][_j]) 
                    _cri = _cc+_cbi1+_cbi2 
                    _crj = _cc+_cbj1+_cbj2 
                    _cr[0][_jdx] = max(_cri,_crj)-1 
                    _csi1 = abs(_ratio[12*_i+1][_j]) 
                    _csi2 = abs(_ratio[12*_i+2][_j]) 
                    _csj1 = abs(_ratio[12*_i+7][_j]) 
                    _csj2 = abs(_ratio[12*_i+8][_j]) 
                    _cs[0][_jdx] = max(_csi1,_csi2,_csj1,_csj2)-1 
        return _bri,_brj,_brc,_cr,_bsi,_bsj,_cs 
    def deformation(__Ho,__RS,_aiy): 
        _dbl = zrs(shape=(1,_nvg)) ; _delta = _dbl.copy() ; _form = _dbl.copy() 
        for _i in range(_nvg): 
            _dbl[0][_i] = 1/15*(_lm[int(_repg[0][_i])]/__Ho[int(_Hn[int(_repg[0][_i])])])-1 
            if _frame == 'x' : 
                if _frame_io == 'i' : _mef = 1 
                else : _mef = 0 
            elif _frame == 'y' : 
                if _frame_io == 'i' : _mef = 3 
                else : _mef = 2 
            else : pass 
            _m0 = _M0[_mef] 
            _delta[0][_i] = 5*_m0*(_lm[int(_repg[0][_i])]**2/(48*_E*_aiy[0][int(_repg[0][_i])]))\
                            -(__RS[4][int(_repg[0][_i])]-__RS[10][int(_repg[0][_i])])\
                            /(16*_E*_aiy[0][int(_repg[0][_i])])*(_lm[int(_repg[0][_i])])**2 
            _form[0][_i] = (_delta[0][_i]/_lm[int(_repg[0][_i])])*300-1 
        return _dbl,_form 
    def inter_story(__d): 
        _deflect = zrs(shape=(1,_nelz)) 
        if _frame == 'x' : 
            _delbyhx = __d[int(_njf[int(6*(_lxyr[0]-1))])][1]/_fht[0] 
            _deflect[0][0] = abs(_delbyhx)*200-1 
            for _i in range(1,_nelz): 
                _du = __d[int(_njf[int(6*(_lxyr[_i]-1))])][1]\
                      -__d[int(_njf[int(6*(_lxyr[_i-1]-1))])][1] 
                _delbyhx = _du/_fht[_i] 
                _deflect[0][_i] = abs(_delbyhx)*200-1 
        elif _frame == 'y' : 
            _delbyhy = __d[int(_njf[int(6*(_lxyr[0]))+1])][1]/_fht[0] 
            _deflect[0][0] = abs(_delbyhy*200)-1 
            for _i in range(1,_nelz): 
                _dv = __d[int(_njf[int(6*(_lxyr[_i]))+1])][1]\
                      -__d[int(_njf[int(6*(_lxyr[_i-1]))+1])][1] 
                _delbyhy = _dv/_fht[_i] 
                _deflect[0][_i] = abs(_delbyhy*200)-1 
        else : pass 
        return _deflect 
    def wt_ratio(__Ho,__Bo,__two,__tfo,__Do,__to): 
        _wid_thick = zrs(shape=(1,2*_nvg)) 
        _wid_c = zrs(shape=(1,_nvc)) 
        for _i in range(_nvg): 
            _H = __Ho[int(_Hn[int(_repg[0][_i])])] 
            _B = __Bo[int(_Bn[int(_repg[0][_i])])] 
            _tw = __two[int(_twn[int(_repg[0][_i])])] 
            _tf = __tfo[int(_tfn[int(_repg[0][_i])])] 
            _btf = _B/2/_tf/9 
            _dtw = (_H-2*_tf)/_tw/60 
            _wid_thick[0][2*_i] = _btf-1 
            _wid_thick[0][2*_i+1] = _dtw-1 
        for _i in range(_nvc): 
            _D = __Do[int(_Dn[int(_repc[0][_i])])] 
            _t = __to[int(_tn[int(_repc[0][_i])])] 
            _bt = _D/_t/33 
            _wid_c[0][_i] = _bt-1 
        return _wid_thick,_wid_c 
    def wt_ratio_L(__Bo,__tfo): 
        _wid_gl = zrs(shape=(1,_nvg)) 
        for _i in range(_nvg): 
            _B = __Bo[int(_Bn[int(_repg[0][_i])])] 
            _tf = __tfo[int(_tfn[int(_repg[0][_i])])] 
            _btf = _B/2/_tf 
            _wid_gl[0][_i] = 4/_btf-1 
        return _wid_gl 
    def thickness(__Do): 
        _column = [] 
        if _frame == 'x' : _nelxy = _nelx 
        elif _frame == 'y' : _nelxy = _nely 
        if _nelxy % 2 == 1 : _idx = int(_nc/2) 
        else : _idx = int(max(_Dn))+1 
        for _i in range(_idx): 
            if _i%_nelz == _nelz-1 : pass 
            else : 
                _b1 = __Do[int(_Dn[_i])] 
                _b2 = __Do[int(_Dn[_i+1])] 
                _column.append(_b2/_b1-1) 
        return np.array(_column) 
    def proof_stress(_zpy): 
        def index(_list,_idx): 
            _index = []  
            for _val in range(len(_list)): 
                if _idx == _list[_val]: 
                    _index.append(_val) 
            return _index 
        def flatten(_): 
            _flt = [] 
            for _i in _ : 
                if hasattr(_i,'__iter__') : _flt += flatten(_i) 
                else : _flt.append(_i) 
            return _flt 
        if _frame == 'x' : _nfj = _nelx+1 ; _ngx = _ng 
        elif _frame == 'y' : _nfj = _nely+1 ; _ngx = 0 
        else : pass 
        _nrps = _nj-2*_nfj 
        _rps = zrs(shape=(1,2*_nrps)) 
        for _i in range(_nfj,_nfj+_nrps): 
            _pci = [] ; _pbix = [] ; _pbiy = [] ; _mem = [] 
            _mem.append(index(list(_js),_i)) ; _mem.append(index(list(_je),_i)) 
            _mem = flatten(_mem) 
            for _j in range(len(_mem)): 
                if _c_g[int(_mem[_j])] == 1 : 
                    _pci.append(_zpy[0][int(_mem[_j])]*_F[int(_jel[int(_mem[_j])])-1]) 
                elif _c_g[int(_mem[_j])] == 2 and _mem[_j] < _ngx : 
                    _pbix.append(_zpy[0][int(_mem[_j])]*_F[int(_jel[int(_mem[_j])])-1]) 
                else : 
                    _pbiy.append(_zpy[0][int(_mem[_j])]*_F[int(_jel[int(_mem[_j])])-1]) 
            _rpc = np.sum(_pci) ; _rpbx = np.sum(_pbix) ; _rpby = np.sum(_pbiy) 
            _rps[0][2*(_i-_nfj)] = 1.5*_rpbx/_rpc-1 
            _rps[0][2*(_i-_nfj)+1] = 1.5*_rpby/_rpc-1 
        return _rps 
    _Ho = _target[_Hp[0]:_Hp[-1]+1] 
    _Bo = _target[_Bp[0]:_Bp[-1]+1] 
    _two = _target[_twp[0]:_twp[-1]+1] 
    _tfo = _target[_tfp[0]:_tfp[-1]+1] 
    _Do = _target[_Dp[0]:_Dp[-1]+1] 
    _to = _target[_tp[0]:_tp[-1]+1] 
    _a,_asx,_asy,_aiy,_aiz,_zy,_zz,_zyf,_zpy,_zpz = datasf(_Ho,_Bo,_two,_tfo,_Do,_to) 
    __s = stifsf(_a,_aiy,_aiz) 
    __s,__f,_r,_sk = suptsf(__s,_f) 
    _d = eqsoln(__s,__f) 
    _arm,_RS,_Mc = rsltsf(_r,_sk,_d) 
    _st,_stc = stress(_RS,_Mc,_a,_asx,_asy,_zy,_zz,_zyf) 
    _gb,_gw = gwidth(_Ho,_Bo) 
    _bri,_brj,_brc,_cr,_bsi,_bsj,_cs  = stress_ratio(_Ho,_Bo,_two,_tfo,_st,_stc,_a,_aiy,_aiz) 
    _dbl,_form = deformation(_Ho,_RS,_aiy) 
    _deflect = inter_story(_d) 
    _wid_thick,_wid_c = wt_ratio(_Ho,_Bo,_two,_tfo,_Do,_to) 
    _wid_gl = wt_ratio_L(_Bo,_tfo) 
    _column = thickness(_Do) 
    _rps = proof_stress(_zpy) 
    return list(-_gb[0]),list(-_gw[0]),list(-_bri[0]),list(-_brj[0]),\
           list(-_brc[0]),list(-_bsi[0]),list(-_bsj[0]),\
           list(-_dbl[0]),list(-_form[0]),list(-_deflect[0]),list(-_wid_thick[0]),\
           list(-_wid_c[0]),list(-_wid_gl[0]),list(-_column),list(-_rps[0]) 
# 一番大事
def objective(_target,_Hp,_Bp,_twp,_tfp,_Dp,_tp,_nm,_ng,_c_g,_Hn,_Bn,_twn,_tfn,_Dn,_tn,_lm,_module): 
    _Ho = _target[_Hp[0]:_Hp[-1]+1] 
    _Bo = _target[_Bp[0]:_Bp[-1]+1] 
    _two = _target[_twp[0]:_twp[-1]+1] 
    _tfo = _target[_tfp[0]:_tfp[-1]+1] 
    _Do = _target[_Dp[0]:_Dp[-1]+1] 
    _to = _target[_tp[0]:_tp[-1]+1] 
    _a = zrs(shape=(1,_nm)) 
    for _i in range(_nm): 
        if _c_g[_i] == 1 : 
            _D = _Do[int(_Dn[_i-_ng])] 
            _t = _to[int(_tn[_i-_ng])] 
            _a[0][_i] = _D**2-(_D-2*_t)**2 
        else : 
            _H = _Ho[int(_Hn[_i])] 
            _B = _Bo[int(_Bn[_i])] 
            _tw = _two[int(_twn[_i])] 
            _tf = _tfo[int(_tfn[_i])] 
            _a[0][_i] = _H*_B-(_B-_tw)*(_H-2*_tf) 
    _fun = np.dot(_a,_lm.T)*_module 
    return _fun[0] 