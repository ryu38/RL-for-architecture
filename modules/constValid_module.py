from time import time 
from numpy import ones as ons 
from numpy import insert as ist 
from .constValid_defs_custom import modelParameter, rf_node_info, nodeCoordinate, memberInfo,\
                            memberLength, directionCosine, bandwidth, sectionGroup,\
                            supportInfo, loadsf, arrangeValue, sectionStandard,\
                            varMember, Constraints, opener, floater, transer, objective 

def const_valid(param, label):
    
    _t0 = time() 
    
    # param = [] ; opener(param,'kuga_param.csv') ; param = floater(param) 
    # label = [] ; opener(label,'kuga_label.csv') ; label = floater(label) 

    param = [param]
    label = [label]
    
    _listIndex = 0 
    # global variables # 
    _nelx = int(param[_listIndex][0]) 
    _nely = int(param[_listIndex][1]) 
    _nelz = int(param[_listIndex][2]) 
    _lx = int(param[_listIndex][3]) 
    _ly = int(param[_listIndex][4]) 
    _lz = int(param[_listIndex][5])  ; _lz_first = _lz+1400 
    _pw = float(param[_listIndex][6]) 
    _frame = 'y' 
    _frame_io = 'i' 
    _E = 2.05*1e5 ; _pr = 0.30 ; _nma = 2 
    _F = [235,325] ; _nlc = 2 ; _Z = 1.0 ; _co = 0.2 
    
    param = param[_listIndex] 
    label = transer(label,_listIndex,_nely,_nelz) 
    
    # Main driver codes _ local variables and scopes # 
    _nj,_nc,_ng,_nsj = modelParameter(_frame,_nelx,_nely,_nelz) 
    _nm = _nc+_ng 
    _jd,_njr,_fnjd,_jf,_njdp,_njef,_njsf,_ndf,_njf = rf_node_info(_frame,_nj,_nelx,_nely,_nelz) 
    _ns6 = 6*_nsj 
    _fht = [_lz for _i in range(_nelz-1)] ; _fht = ist(_fht,0,_lz_first) 
    _aj = ons(shape=(_nm,1)) 
    _x,_y,_z,_xe,_ye,_ze,_xi,_yi,_zi = nodeCoordinate(_frame,_nj,_nelx,_nely,_nelz,_lx,_ly,_lz,_lz_first) 
    _js,_je,_jel,_c_g = memberInfo(_frame,_nm,_nelx,_nely,_nelz) 
    _lm = memberLength(_nm,_x,_y,_z,_js,_je) 
    _cyl = directionCosine(_nm,_js,_je,_x,_y,_z) 
    _nb = bandwidth(_nm,_njsf,_njef,_js,_je) 
    _Hn,_Bn,_twn,_tfn,_Dn,_tn = sectionGroup(_frame,_nelx,_nely,_nelz) 
    _ns,_isup,_pd = supportInfo(_nsj) 
    _xr = _x.copy() ; _yr = _y.copy() 
    for _i in range(len(_ns)): 
        _xr[_ns[_i]] = 0 ; _yr[_ns[_i]] = 0 
    _seismicForce,_memberEnd,_M0,_f,_lxyr = loadsf(_fht,_xi,_yi,_ndf,_c_g,_frame,_njf,_ng,_frame_io,\
                                                   _x,_y,_z,_je,_js,_cyl,_pw,_Z,_co,_lx,_ly,_nlc,_nelx,_nely,_nelz) 
    _Hp,_Bp,_twp,_tfp,_Dp,_tp = arrangeValue(_Hn,_Bn,_twn,_tfn,_Dn,_tn) 
    _He,_Be,_twe,_tfe,_De,_te = sectionStandard() 
    _nvg = len(_twp) ; _nvc = len(_Dp) ; _nvars = _nvg+_nvc 
    _repg,_repc = varMember(_nvg,_nvc,_twn,_Dn) 
    # 制約検定
    gb,gw,bri,brj,brc,bsi,bsj,dbl,form,deflect,wid_thick,wid_c,wid_gl,column,rps\
        = Constraints(label,_nm,_ng,_nc,_frame,_frame_io,_c_g,_Hn,_Bn,_twn,_tfn,_Dn,_tn,\
                      _ndf,_nb,_je,_js,_x,_y,_z,_cyl,_aj,_E,_pr,_xr,_yr,_njf,_ns6,_nlc,_nsj,\
                      _ns,_isup,_pd,_memberEnd,_M0,_nelz,_nma,_F,_jel,_lm,_Hp,_Bp,_twp,_tfp,_Dp,_tp,_f,\
                      _nvg,_nvc,_repg,_repc,_lxyr,_fht,_nelx,_nely,_nj) 
    _module = 1e-6 
    objectiveFunction = objective(label,_Hp,_Bp,_twp,_tfp,_Dp,_tp,_nm,_ng,_c_g,_Hn,_Bn,_twn,_tfn,_Dn,_tn,_lm,_module) 

    return [gb,gw,bri,brj,brc,bsi,bsj,dbl,form,deflect,wid_thick,wid_c,wid_gl,column,rps]
    