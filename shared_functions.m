%% shared_functions.m 
function sf = shared_functions()
    sf.build_params = @build_params;
    sf.calc_ni = @calc_ni;
    sf.calc_pf = @calc_pf;
    sf.obj_det = @obj_det;
    sf.con_det = @con_det;
    sf.con_master = @con_master;
    sf.solve_moment_DRCCO_t = @solve_moment_DRCCO_t;
    sf.Psi_k = @Psi_k;
    sf.bw_all = @bw_all;
    sf.bw_obj = @bw_obj;
    % --- M4 双边 ---
    sf.dro_obj_analytical = @dro_obj_analytical;
    sf.obj_dro_an = @obj_dro_an;
    sf.solve_cp_analytical = @solve_cp_analytical;
    sf.eval_cc_analytical = @eval_cc_analytical;
    % --- M3 单边 ---
    sf.obj_dro_an_oneside = @obj_dro_an_oneside;
    sf.dro_obj_oneside = @dro_obj_oneside;
    sf.solve_cp_oneside = @solve_cp_oneside;
    sf.eval_cc_oneside = @eval_cc_oneside;
    % --- fmincon 子问题求解器 ---
    sf.solve_kde_sub_bilateral_fmincon = @solve_kde_sub_bilateral_fmincon;
    sf.solve_kde_sub_oneside_fmincon = @solve_kde_sub_oneside_fmincon;
    sf.solve_dro_obj_fmincon = @solve_dro_obj_fmincon;
end

%% ==================== 参数构建 ====================
function pt = build_params(t, P_load_t, Q_load_t, P_DG_f, xi_t, ...
        nb, nbr, nl, nd, ns, nseg, lbs, dbs, fb, tb, WW, Smx, ...
        as, bs, cs, v0, vmn, vmx, tth, eps_v, tau_v, tau_o, e1, e2, e3, bMVA, ...
        Rm, Xm, AVL, AVD, ABL, ABD, AOL, AOD, BCi, nvc, nbc, id, Sigma_diag, DG_types)
    pt.num_bus=nb; pt.num_branch=nbr; pt.num_load=nl; pt.num_DG=nd;
    pt.num_samples=ns; pt.num_segments=nseg;
    pt.load_buses=lbs; pt.DG_buses=dbs; pt.from_bus=fb; pt.to_bus=tb;
    pt.W=WW; pt.S_max_br=Smx;
    pt.a_s=as; pt.b_s=bs; pt.c_s=cs;
    pt.V0=v0; pt.V_min=vmn; pt.V_max=vmx; pt.tan_theta=tth;
    pt.epsilon=eps_v; pt.tau=tau_v; pt.tau_obj=tau_o;
    pt.eta1=e1; pt.eta2=e2; pt.eta3=e3; pt.baseMVA=bMVA;
    pt.P_L=P_load_t(:,t); pt.Q_L=Q_load_t(:,t); pt.P_DG=P_DG_f(:,t);
    pt.R_mat=Rm; pt.X_mat=Xm; pt.xi=xi_t;
    pt.A_V_load=AVL; pt.A_V_DG=AVD;
    pt.A_BC_load=ABL; pt.A_BC_DG=ABD;
    pt.A_obj_load=AOL; pt.A_obj_DG=AOD;
    pt.BC_info=BCi; pt.n_V_constr=nvc; pt.n_BC_constr=nbc;
    pt.idx=id; pt.K=nvc+nbc;
    pt.Sigma_diag=Sigma_diag; pt.Sigma=diag(Sigma_diag);
    pt.DG_types=DG_types;
end

%% ==================== 基础工具 ====================
function [pn,qn]=calc_ni(x,p)
    pn=zeros(p.num_bus,1); qn=zeros(p.num_bus,1);
    for i=1:p.num_load
        b=p.load_buses(i);
        pn(b)=p.P_L(b)-x(p.idx.pcut(i));
        qn(b)=p.Q_L(b)-x(p.idx.qcut(i));
    end
    for i=1:p.num_DG
        b=p.DG_buses(i);
        pn(b)=pn(b)-p.P_DG(i)*(1-x(p.idx.lambda(i)));
    end
end

function [V,Pb,Qb,Sb]=calc_pf(x,p)
    [pn,qn]=calc_ni(x,p);
    V=zeros(p.num_bus,1); V(1)=p.V0;
    for i=2:p.num_bus
        V(i)=p.V0-sum(pn(2:end).*p.R_mat(i,2:end)'+qn(2:end).*p.X_mat(i,2:end)');
    end
    Pb=zeros(p.num_branch,1); Qb=zeros(p.num_branch,1);
    for br=1:p.num_branch
        ds=p.W{p.to_bus(br)};
        Pb(br)=sum(pn(ds)); Qb(br)=sum(qn(ds));
    end
    Sb=sqrt(Pb.^2+Qb.^2);
end

function c=obj_det(x,p)
    c=p.eta1*x(p.idx.pplus)-p.eta2*x(p.idx.pminus)+p.eta3*sum(x(p.idx.pcut));
end

function [c,ceq]=con_det(x,p)
    [pn,qn]=calc_ni(x,p);
    ceq=x(p.idx.pplus)-x(p.idx.pminus)-sum(pn(2:end));
    c=[];
    for i=2:p.num_bus
        Vi=p.V0-sum(pn(2:end).*p.R_mat(i,2:end)'+qn(2:end).*p.X_mat(i,2:end)');
        c=[c;Vi-p.V_max;p.V_min-Vi];
    end
    for br=1:p.num_branch
        ds=p.W{p.to_bus(br)};
        Pbv=sum(pn(ds)); Qbv=sum(qn(ds));
        for s=1:p.num_segments/2
            c=[c; p.a_s(s)*Pbv+p.b_s(s)*Qbv-p.c_s(s)*p.S_max_br(br)];
            c=[c;-p.a_s(s)*Pbv-p.b_s(s)*Qbv-p.c_s(s)*p.S_max_br(br)];
        end
    end
end

function [c,ceq]=con_master(x,p)
    [pn,~]=calc_ni(x,p);
    ceq=x(p.idx.pplus)-x(p.idx.pminus)-sum(pn(2:end));
    c=[];
end

%% ==================== M2 Moment-DRCCO  ====================
function [x_opt,fval,info] = solve_moment_DRCCO_t(p)
    id = p.idx;
    nL=p.num_load; nD=p.num_DG; nxi=nL+nD;
    ep=p.epsilon;
    sq_sig = sqrt(p.Sigma_diag);

    pc  = sdpvar(nL,1); qc  = sdpvar(nL,1);
    lam = sdpvar(nD,1); pp  = sdpvar(1,1); pm  = sdpvar(1,1);
    obj = p.eta1*pp - p.eta2*pm + p.eta3*sum(pc);

    C = [pc>=0, qc>=0, lam>=0, lam<=1, pp>=0, pm>=0, pp<=10, pm<=10];
    for i=1:nL
        C=[C, pc(i)<=p.P_L(p.load_buses(i)), qc(i)<=p.Q_L(p.load_buses(i))];
    end

    pn_s = cell(p.num_bus,1); qn_s = cell(p.num_bus,1);
    for i=1:p.num_bus, pn_s{i}=0; qn_s{i}=0; end
    for i=1:nL
        b=p.load_buses(i); pn_s{b}=p.P_L(b)-pc(i); qn_s{b}=p.Q_L(b)-qc(i);
    end
    for i=1:nD
        b=p.DG_buses(i); pn_s{b}=pn_s{b}-p.P_DG(i)*(1-lam(i));
    end
    sp=0; for i=2:p.num_bus, sp=sp+pn_s{i}; end
    C = [C, pp-pm==sp];

    gV = (p.V_max-p.V_min)/2; Vc = (p.V_max+p.V_min)/2;
    y1V = sdpvar(p.n_V_constr,1); y2V = sdpvar(p.n_V_constr,1);
    for k=1:p.n_V_constr
        ib=k+1; Vb=p.V0;
        for j=2:p.num_bus
            Vb=Vb-(pn_s{j}*p.R_mat(ib,j)+qn_s{j}*p.X_mat(ib,j));
        end
        beta_k = Vb - Vc;
        w_vec = sdpvar(nxi,1);
        for jj=1:nL, w_vec(jj) = p.A_V_load(k,jj)*sq_sig(jj); end
        for jj=1:nD, w_vec(nL+jj) = p.A_V_DG(k,jj)*(1-lam(jj))*sq_sig(nL+jj); end
        C = [C, y1V(k)>=0, y2V(k)>=0, y2V(k)<=gV];
        C = [C, cone([y1V(k); w_vec], sqrt(ep)*(gV-y2V(k)))];
        C = [C, beta_k <= y1V(k)+y2V(k), -beta_k <= y1V(k)+y2V(k)];
    end

    y1B = sdpvar(p.n_BC_constr,1); y2B = sdpvar(p.n_BC_constr,1);
    for kk=1:p.n_BC_constr
        br=p.BC_info(kk,1); si=p.BC_info(kk,2);
        gBC = p.c_s(si)*p.S_max_br(br);
        ds=p.W{p.to_bus(br)}; Pb=0; Qb=0;
        for dd=1:length(ds), Pb=Pb+pn_s{ds(dd)}; Qb=Qb+qn_s{ds(dd)}; end
        beta_bc = p.a_s(si)*Pb + p.b_s(si)*Qb;
        w_bc = sdpvar(nxi,1);
        for jj=1:nL, w_bc(jj) = p.A_BC_load(kk,jj)*sq_sig(jj); end
        for jj=1:nD, w_bc(nL+jj) = p.A_BC_DG(kk,jj)*(1-lam(jj))*sq_sig(nL+jj); end
        C = [C, y1B(kk)>=0, y2B(kk)>=0, y2B(kk)<=gBC];
        C = [C, cone([y1B(kk); w_bc], sqrt(ep)*(gBC-y2B(kk)))];
        C = [C, beta_bc<=y1B(kk)+y2B(kk), -beta_bc<=y1B(kk)+y2B(kk)];
    end

    opts = sdpsettings('solver','mosek','verbose',0,'debug',0);
    sol = optimize(C, obj, opts);
    if sol.problem ~= 0
        opts2 = sdpsettings('solver','sedumi','verbose',0);
        sol = optimize(C, obj, opts2);
        if sol.problem ~= 0
            opts3 = sdpsettings('solver','sdpt3','verbose',0);
            sol = optimize(C, obj, opts3);
        end
    end

    x_opt = zeros(id.nx,1);
    x_opt(id.pcut) = max(0, value(pc));
    x_opt(id.qcut) = max(0, value(qc));
    x_opt(id.lambda) = max(0, min(1, value(lam)));
    x_opt(id.pplus) = max(0, value(pp));
    x_opt(id.pminus) = max(0, value(pm));
    fval = value(obj);
    info = sol;
end

%% ==================== KDE 核心 ====================
function Psi = Psi_k(u, h)
    if h<1e-12, Psi=max(u,0); return; end
    z=u/h; z=max(-30,min(30,z));
    Psi=u*(0.5*(1+erf(z/sqrt(2))))+h*exp(-0.5*z^2)/sqrt(2*pi);
end

function [hV,hB]=bw_all(lam,p)
    MM=p.num_samples; hV=zeros(p.n_V_constr,1); hB=zeros(p.n_BC_constr,1);
    use_kde=isfield(p,'use_kde')&&p.use_kde;
    if ~use_kde, return; end
    for k=1:p.n_V_constr
        aL=p.A_V_load(k,:)'; aD=p.A_V_DG(k,:)'.*(1-lam);
        Y=aL'*p.xi(1:p.num_load,:)+aD'*p.xi(p.num_load+1:end,:);
        sg=std(Y); if sg>1e-10, hV(k)=1.06*sg*MM^(-0.2); end
    end
    for k=1:p.n_BC_constr
        aL=p.A_BC_load(k,:)'; aD=p.A_BC_DG(k,:)'.*(1-lam);
        Y=aL'*p.xi(1:p.num_load,:)+aD'*p.xi(p.num_load+1:end,:);
        sg=std(Y); if sg>1e-10, hB(k)=1.06*sg*MM^(-0.2); end
    end
end

function ho=bw_obj(lam,p)
    MM=p.num_samples;
    aL=p.A_obj_load'; aD=p.A_obj_DG'.*(1-lam);
    Y=aL'*p.xi(1:p.num_load,:)+aD'*p.xi(p.num_load+1:end,:);
    sg=std(Y); if sg<1e-10, ho=0; else, ho=1.06*sg*MM^(-0.2); end
end

%% ==================== 子问题 ====================
function [fval, y_opt] = solve_kde_sub_bilateral_fmincon(Y, gamma, h, eps_c, tau, M)
    Y = Y(:)';
    nu_min = 1e-4;

    eta0 = 0;
    d0 = compute_d_bilateral(eta0, Y, gamma, h, M) / eps_c;
    mu0 = median(d0);
    nu0 = max(std(d0 - mu0), 0.1);
    z0 = [eta0; nu0; mu0];
    lb = [-inf; nu_min; -inf];
    ub = [ inf; inf;    inf];

    opts = optimoptions('fmincon','Display','off','Algorithm','sqp',...
        'MaxIterations',200,'MaxFunctionEvaluations',2000,...
        'OptimalityTolerance',1e-8,'ConstraintTolerance',1e-8,...
        'StepTolerance',1e-12);

    obj_fn = @(z) obj_3d_bilateral(z, Y, gamma, h, eps_c, tau, M);

    try
        [y_opt, fval] = fmincon(obj_fn, z0, [], [], [], [], lb, ub, [], opts);
    catch
        y_opt = z0; fval = obj_3d_bilateral(z0, Y, gamma, h, eps_c, tau, M);
    end
end

function [fval, y_opt] = solve_kde_sub_oneside_fmincon(Y, gamma, h, eps_c, tau, M)
    Y = Y(:)';
    nu_min = 1e-4;

    eta0 = 0;
    d0 = compute_d_oneside(eta0, Y, gamma, h, M) / eps_c;
    mu0 = median(d0);
    nu0 = max(std(d0 - mu0), 0.1);
    z0 = [eta0; nu0; mu0];

    lb = [-inf; nu_min; -inf];
    ub = [ inf; inf;    inf];

    opts = optimoptions('fmincon','Display','off','Algorithm','sqp',...
        'MaxIterations',200,'MaxFunctionEvaluations',2000,...
        'OptimalityTolerance',1e-8,'ConstraintTolerance',1e-8,...
        'StepTolerance',1e-12);

    obj_fn = @(z) obj_3d_oneside(z, Y, gamma, h, eps_c, tau, M);

    try
        [y_opt, fval] = fmincon(obj_fn, z0, [], [], [], [], lb, ub, [], opts);
    catch
        y_opt = z0; fval = obj_3d_oneside(z0, Y, gamma, h, eps_c, tau, M);
    end
end

function [fval, y_opt] = solve_dro_obj_fmincon(g1, h_obj, eta1, eta2, tau, M)
    g1 = g1(:)';
    nu_min = 1e-4;
    d = zeros(1, M);
    for m = 1:M
        d(m) = (eta1-eta2)/2 * (Psi_k(-g1(m), h_obj) + Psi_k(g1(m), h_obj)) ...
             + (eta1+eta2)/2 * g1(m);
    end
    mu0 = median(d);
    nu0 = max(std(d - mu0), 0.1);
    z0 = [nu0; mu0];

    lb = [nu_min; -inf];
    ub = [inf;     inf];

    opts = optimoptions('fmincon','Display','off','Algorithm','sqp',...
        'MaxIterations',200,'MaxFunctionEvaluations',1500,...
        'OptimalityTolerance',1e-8,'ConstraintTolerance',1e-8,...
        'StepTolerance',1e-12);

    obj_fn = @(z) obj_2d_dro_obj(z, d, tau, M);

    try
        [y_opt, fval] = fmincon(obj_fn, z0, [], [], [], [], lb, ub, [], opts);
    catch
        y_opt = z0; fval = obj_2d_dro_obj(z0, d, tau, M);
    end
end

function d = compute_d_bilateral(eta, Y, gamma, h, M)
    C = gamma + eta;
    Cp = max(C, 0); Cm = max(-C, 0);
    d = zeros(1, M);
    for m = 1:M
        d(m) = Psi_k(-Y(m) - Cp, h) + Psi_k(Y(m) - Cp, h) + Cm;
    end
end

function d = compute_d_oneside(eta, Y, gamma, h, M)
    d = zeros(1, M);
    for m = 1:M
        d(m) = Psi_k(Y(m) - gamma - eta, h);
    end
end

function f = obj_3d_bilateral(z, Y, gamma, h, eps_c, tau, M)
    eta = z(1); nu = z(2); mu = z(3);
    d = compute_d_bilateral(eta, Y, gamma, h, M) / eps_c;
    t = d - mu + 2*nu;
    tp = max(t, 0);
    f = eta + nu*(tau - 1) + mu + sum(tp.^2) / (4*M*nu);
end

function f = obj_3d_oneside(z, Y, gamma, h, eps_c, tau, M)
    eta = z(1); nu = z(2); mu = z(3);
    d = compute_d_oneside(eta, Y, gamma, h, M) / eps_c;
    t = d - mu + 2*nu;
    tp = max(t, 0);
    f = eta + nu*(tau - 1) + mu + sum(tp.^2) / (4*M*nu);
end

function f = obj_2d_dro_obj(z, d, tau, M)
    nu = z(1); mu = z(2);
    t = d - mu + 2*nu;
    tp = max(t, 0);
    f = nu*(tau - 1) + mu + sum(tp.^2) / (4*M*nu);
end

%% ====================================================================
%%                M4: 双边 KDE-DRCCO
%% ====================================================================

function dv = dro_obj_analytical(lam, p)
    T = p.num_samples; tv = p.tau_obj;
    use_kde = isfield(p, 'use_kde') && p.use_kde;
    aL = p.A_obj_load'; aD = p.A_obj_DG'.*(1-lam);
    g1 = aL'*p.xi(1:p.num_load,:) + aD'*p.xi(p.num_load+1:end,:);
    if use_kde
        sig = std(g1); ho = max(1.06*sig*T^(-0.2), 0);
    else
        ho = 0;
    end
    [dv, ~] = solve_dro_obj_fmincon(g1, ho, p.eta1, p.eta2, tv, T);
end

function cost = obj_dro_an(x, p)
    id = p.idx;
    cost = p.eta1*x(id.pplus) - p.eta2*x(id.pminus) + p.eta3*sum(x(id.pcut));
    cost = cost + dro_obj_analytical(x(id.lambda), p);
end

% --------- 双边: cutting-plane 主循环 ---------
function [x_opt, fval, ni] = solve_cp_analytical(x_init, p, lb, ub, opts_f, cg)
    [x_opt, fval, ni] = cutplane_loop(x_init, p, lb, ub, opts_f, cg, 'bilateral');
end

function [Fk, gk, vio] = eval_cc_analytical(x, p, cg)
    id = p.idx; nx = id.nx; KK = p.K; T = p.num_samples;
    ec = p.epsilon; tv = p.tau;
    Fk = zeros(KK, 1); gk = zeros(nx, KK); vio = false(KK, 1);
    lam = x(id.lambda);
    [pn, qn] = calc_ni(x, p);
    Vb = zeros(p.num_bus, 1); Vb(1) = p.V0;
    for i = 2:p.num_bus
        Vb(i) = p.V0 - sum(pn(2:end).*p.R_mat(i,2:end)' + qn(2:end).*p.X_mat(i,2:end)');
    end
    gV = (p.V_max - p.V_min)/2; Vc = (p.V_max + p.V_min)/2;
    Pbr = zeros(p.num_branch,1); Qbr = zeros(p.num_branch,1);
    for br = 1:p.num_branch
        ds = p.W{p.to_bus(br)};
        Pbr(br) = sum(pn(ds)); Qbr(br) = sum(qn(ds));
    end
    kg = 0;
    for k = 1:p.n_V_constr
        kg = kg + 1;
        bk_val = Vb(k+1) - Vc; h_val = p.hV(k);
        aL = p.A_V_load(k,:)'; aD = p.A_V_DG(k,:)'.*(1-lam);
        Y = aL'*p.xi(1:p.num_load,:) + aD'*p.xi(p.num_load+1:end,:) + bk_val;
        [F_val, y_star] = solve_kde_sub_bilateral_fmincon(Y, gV, h_val, ec, tv, T);
        Fk(kg) = F_val;
        if F_val > 1e-6
            vio(kg) = true;
            d_V = pack_dV(p, k);
            res = cg.f_V_bi('x', x, 'y', y_star, 'd_V', d_V, 'h', h_val);
            gk(:,kg) = full(res.grad);
        end
    end
    for k = 1:p.n_BC_constr
        kg = kg + 1;
        br = p.BC_info(k,1); si = p.BC_info(k,2);
        bk_val = p.a_s(si)*Pbr(br) + p.b_s(si)*Qbr(br);
        gam = p.c_s(si)*p.S_max_br(br); h_val = p.hB(k);
        aL = p.A_BC_load(k,:)'; aD = p.A_BC_DG(k,:)'.*(1-lam);
        Y = aL'*p.xi(1:p.num_load,:) + aD'*p.xi(p.num_load+1:end,:) + bk_val;
        [F_val, y_star] = solve_kde_sub_bilateral_fmincon(Y, gam, h_val, ec, tv, T);
        Fk(kg) = F_val;
        if F_val > 1e-6
            vio(kg) = true;
            d_BC = pack_dBC(p, k);
            res = cg.f_BC_bi('x', x, 'y', y_star, 'd_BC', d_BC, ...
                'gamma', gam, 'a', p.a_s(si), 'b', p.b_s(si), 'h', h_val);
            gk(:,kg) = full(res.grad);
        end
    end
end

%% ====================================================================
%%       M3: 单边 KDE-DRCCO (eps/2)
%% ====================================================================

function dv = dro_obj_oneside(lam, p)
    dv = dro_obj_analytical(lam, p);
end

function cost = obj_dro_an_oneside(x, p)
    id = p.idx;
    cost = p.eta1*x(id.pplus) - p.eta2*x(id.pminus) + p.eta3*sum(x(id.pcut));
    cost = cost + dro_obj_oneside(x(id.lambda), p);
end

% --------- 单边: cutting-plane 主循环 ---------
function [x_opt, fval, ni] = solve_cp_oneside(x_init, p, lb, ub, opts_f, cg)
    [x_opt, fval, ni] = cutplane_loop(x_init, p, lb, ub, opts_f, cg, 'oneside');
end

function [Fk, gk, vio] = eval_cc_oneside(x, p, cg)
    id = p.idx; nx = id.nx; T = p.num_samples;
    ec_half = p.epsilon / 2;
    tv = p.tau;
    KK_total = 2 * (p.n_V_constr + p.n_BC_constr);
    Fk = zeros(KK_total, 1); gk = zeros(nx, KK_total); vio = false(KK_total, 1);
    lam = x(id.lambda);
    [pn, qn] = calc_ni(x, p);
    Vb = zeros(p.num_bus, 1); Vb(1) = p.V0;
    for i = 2:p.num_bus
        Vb(i) = p.V0 - sum(pn(2:end).*p.R_mat(i,2:end)' + qn(2:end).*p.X_mat(i,2:end)');
    end
    Pbr = zeros(p.num_branch,1); Qbr = zeros(p.num_branch,1);
    for br = 1:p.num_branch
        ds = p.W{p.to_bus(br)};
        Pbr(br) = sum(pn(ds)); Qbr(br) = sum(qn(ds));
    end
    kg = 0;
    for k = 1:p.n_V_constr
        h_val = p.hV(k);
        aL = p.A_V_load(k,:)'; aD = p.A_V_DG(k,:)'.*(1-lam);
        Y_raw = aL'*p.xi(1:p.num_load,:) + aD'*p.xi(p.num_load+1:end,:);

        Y_up = Y_raw + Vb(k+1);
        Y_lo = -(Y_raw + Vb(k+1));

        kg = kg + 1;
        [F_up, y_up] = solve_kde_sub_oneside_fmincon(Y_up, p.V_max, h_val, ec_half, tv, T);
        Fk(kg) = F_up;
        if F_up > 1e-6
            vio(kg) = true;
            d_V = pack_dV(p, k);
            res = cg.f_V_os('x', x, 'y', y_up, 'd_V', d_V, ...
                'gamma', p.V_max, 'h', h_val, 'side', +1);
            gk(:,kg) = full(res.grad);
        end

        kg = kg + 1;
        [F_lo, y_lo] = solve_kde_sub_oneside_fmincon(Y_lo, -p.V_min, h_val, ec_half, tv, T);
        Fk(kg) = F_lo;
        if F_lo > 1e-6
            vio(kg) = true;
            d_V = pack_dV(p, k);
            res = cg.f_V_os('x', x, 'y', y_lo, 'd_V', d_V, ...
                'gamma', -p.V_min, 'h', h_val, 'side', -1);
            gk(:,kg) = full(res.grad);
        end
    end
    for k = 1:p.n_BC_constr
        br = p.BC_info(k,1); si = p.BC_info(k,2); h_val = p.hB(k);
        bk_val = p.a_s(si)*Pbr(br) + p.b_s(si)*Qbr(br);
        gam = p.c_s(si)*p.S_max_br(br);
        aL = p.A_BC_load(k,:)'; aD = p.A_BC_DG(k,:)'.*(1-lam);
        Y_raw = aL'*p.xi(1:p.num_load,:) + aD'*p.xi(p.num_load+1:end,:);

        Y_up = Y_raw + bk_val;
        Y_lo = -(Y_raw + bk_val);

        kg = kg + 1;
        [F_up, y_up] = solve_kde_sub_oneside_fmincon(Y_up, gam, h_val, ec_half, tv, T);
        Fk(kg) = F_up;
        if F_up > 1e-6
            vio(kg) = true;
            d_BC = pack_dBC(p, k);
            res = cg.f_BC_os('x', x, 'y', y_up, 'd_BC', d_BC, ...
                'gamma', gam, 'a', p.a_s(si), 'b', p.b_s(si), ...
                'h', h_val, 'side', +1);
            gk(:,kg) = full(res.grad);
        end

        kg = kg + 1;
        [F_lo, y_lo] = solve_kde_sub_oneside_fmincon(Y_lo, gam, h_val, ec_half, tv, T);
        Fk(kg) = F_lo;
        if F_lo > 1e-6
            vio(kg) = true;
            d_BC = pack_dBC(p, k);
            res = cg.f_BC_os('x', x, 'y', y_lo, 'd_BC', d_BC, ...
                'gamma', gam, 'a', p.a_s(si), 'b', p.b_s(si), ...
                'h', h_val, 'side', -1);
            gk(:,kg) = full(res.grad);
        end
    end
end

function d_V = pack_dV(p, k)
    d_V = zeros(p.num_load + p.num_DG + 2*(p.num_bus-1), 1);
    d_V(1:p.num_load) = p.A_V_load(k,:)';
    d_V(p.num_load+1:p.num_load+p.num_DG) = p.A_V_DG(k,:)';
    d_V(p.num_load+p.num_DG+1:p.num_load+p.num_DG+p.num_bus-1) = p.R_mat(k+1,2:end)';
    d_V(p.num_load+p.num_DG+p.num_bus:end) = p.X_mat(k+1,2:end)';
end

function d_BC = pack_dBC(p, k)
    mask = zeros(p.num_bus-1, 1);
    br = p.BC_info(k, 1);
    ds = p.W{p.to_bus(br)};
    for jj = 1:length(ds)
        if ds(jj) >= 2, mask(ds(jj)-1) = 1; end
    end
    d_BC = zeros(p.num_load + p.num_DG + (p.num_bus-1), 1);
    d_BC(1:p.num_load) = p.A_BC_load(k,:)';
    d_BC(p.num_load+1:p.num_load+p.num_DG) = p.A_BC_DG(k,:)';
    d_BC(p.num_load+p.num_DG+1:end) = mask;
end

%% ====================================================================
%%   cutting-plane 主循环
%% ====================================================================
function [x_opt, fval, ni] = cutplane_loop(x_init, p, lb, ub, opts_f, cg, mode)
% mode = 'bilateral' (M4) 或 'oneside' (M3)

    if strcmp(mode, 'bilateral')
        obj_fn  = @(xx) obj_dro_an(xx, p);
        eval_fn = @(xx, pp) eval_cc_analytical(xx, pp, cg);
    else
        obj_fn  = @(xx) obj_dro_an_oneside(xx, p);
        eval_fn = @(xx, pp) eval_cc_oneside(xx, pp, cg);
    end

    % ---- 主循环参数 ----
    mi          = 60;        % 最大迭代
    eps_feas    = 1e-4;      % 可行性容差
    stagnate_max = 3;        % 连续多少次无进展触发应急
    move_eps    = 1e-8;      % 判定 "x 没动" 的阈值

    % ---- 主问题 fmincon 选项 ----
    opts_main = opts_f;
    opts_sqp  = optimoptions('fmincon','Display','off','Algorithm','sqp',...
        'MaxIterations',500,'MaxFunctionEvaluations',5e4,...
        'OptimalityTolerance',1e-6,'ConstraintTolerance',1e-6);

    x_opt = x_init;
    Ac = []; bc = [];
    fval = obj_fn(x_init);
    ni = 0;

    stagnate_cnt = 0;
    last_x = x_opt;

    for iter = 1:mi
        ni = iter;

        % ---- 求解主问题 ----
        xn = x_opt; fvn = fval; ef = -99;
        ok = false;
        try
            [xn, fvn, ef] = fmincon(obj_fn, x_opt, ...
                Ac, bc, [], [], lb, ub, @(xx) con_master(xx, p), opts_main);
            ok = all(isfinite(xn));
        catch
            ok = false;
        end
        % 主问题失败或返回 -2, 备用 sqp 重试
        if ~ok || ef == -2
            try
                [xn2, fvn2, ef2] = fmincon(obj_fn, x_opt, ...
                    Ac, bc, [], [], lb, ub, @(xx) con_master(xx, p), opts_sqp);
                if all(isfinite(xn2)) && ef2 ~= -2
                    xn = xn2; fvn = fvn2; ef = ef2; ok = true;
                end
            catch
            end
        end

        % ---- 接受准则 ----
        accept = false;
        if ok
            if iter == 1
                % 首次迭代: 只要可行就接受
                accept = true;
            else
                % 后续: 目标改进 或 ef 良好(>0) 就接受
                if (fvn <= fval + 1e-6) || (ef > 0 && all(isfinite(xn)))
                    accept = true;
                end
            end
        end

        if accept
            x_opt = xn;
            fval  = fvn;
        end

        % ---- 评估约束违反 ----
        lam_k = x_opt(p.idx.lambda);
        [hV, hB] = bw_all(lam_k, p);
        p.hV = hV; p.hB = hB;
        [Fk, gk, vio] = eval_fn(x_opt, p);
        mF = max(Fk); nv = sum(vio);

        if nv == 0 || mF <= eps_feas
            return;
        end

        % ---- 进展检测 ----
        if norm(x_opt - last_x, inf) < move_eps
            stagnate_cnt = stagnate_cnt + 1;
        else
            stagnate_cnt = 0;
        end
        last_x = x_opt;

        % ---- 应急: 主问题不动 ----
        if stagnate_cnt >= stagnate_max
            % 取最违反约束的负梯度方向, 投影到 bounds, 用 backtracking line search
            [~, kw] = max(Fk);
            dir = -gk(:, kw);
            if norm(dir, inf) < 1e-12
                break;  % 梯度退化, 无法推进, 提前结束
            end
            dir = dir / max(1, norm(dir, inf));
            alpha = 1.0;
            improved = false;
            for ls = 1:20
                x_try = max(lb, min(ub, x_opt + alpha * dir));
                % 投影: 修正 con_master 等式约束 (功率平衡)
                [pn_t, ~] = calc_ni(x_try, p);
                imb = sum(pn_t(2:end));
                x_try(p.idx.pplus)  = max(0, imb);
                x_try(p.idx.pminus) = max(0, -imb);
                % 重新评估 maxFk
                lam_t = x_try(p.idx.lambda);
                [hV_t, hB_t] = bw_all(lam_t, p);
                p2 = p; p2.hV = hV_t; p2.hB = hB_t;
                [Fk_t, ~, ~] = eval_fn(x_try, p2);
                if max(Fk_t) < mF * 0.999
                    x_opt = x_try; fval = obj_fn(x_opt);
                    improved = true;
                    break;
                end
                alpha = alpha * 0.5;
            end
            stagnate_cnt = 0;
            if ~improved
                break;  % 应急也无法推进 -> 退出
            end
            % 应急成功后, 进入下一轮再加 cut
            continue;
        end

        % ---- 添加 cut: ak * x <= bk ----
        vv = find(vio);
        n_added = 0;
        for k = vv'
            ak = gk(:,k)';
            bk = -Fk(k) + ak * x_opt;
            if any(~isfinite(ak)) || ~isfinite(bk), continue; end
            nr = norm(ak, inf);
            if nr < 1e-12, continue; end
            ak = ak / nr;
            bk = bk / nr;
            Ac = [Ac; ak]; bc = [bc; bk];
            n_added = n_added + 1;
        end

        % ---- 周期性清理: 丢弃在 x_opt 处不紧的 cut (slack > 1e-3) ----
        if mod(iter, 5) == 0 && size(Ac, 1) > 200
            slack = bc - Ac * x_opt;   % >0 表示松弛
            keep  = slack < 1e-3;
            % 总是保留最近 100 个 cut 
            keep(max(1, end-100):end) = true;
            Ac = Ac(keep, :); bc = bc(keep);
        end
        % 上限
        if size(Ac, 1) > 3000
            Ac = Ac(end-2999:end, :);
            bc = bc(end-2999:end);
        end
    end
end