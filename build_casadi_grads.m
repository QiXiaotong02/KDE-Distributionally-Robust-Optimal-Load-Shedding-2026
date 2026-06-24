function cg = build_casadi_grads(params)
% build_casadi_grads - 构建 CasADi 自动微分函数 (每时段调用一次)
%
% 本函数为所有 (电压/支路) x (双边/单边) 四类约束, 以及 DRO 目标
%
% 所有约束共用统一的子问题形式:
%     c_k(x, y) = eta + nu(tau-1) + mu + 1/(4M nu) * sum_m ([d_m/eps - mu + 2nu]_+)^2
% 其中 eta=y(1), nu=y(2), mu=y(3)
%
% 对双边: d_m = Psi(-Y_m-C+, h) + Psi(Y_m-C+, h) + C-, C = gamma + eta
% 对单边: d_m = Psi(side*Y_m - gamma - eta, h), side=+1(上界) / -1(下界)
%
% 输出结构体 cg 包含 5 个 casadi.Function:
%   cg.f_V_bi (x, y, d_V, h)                 -> (c, grad_x)
%   cg.f_BC_bi(x, y, d_BC, gamma, a, b, h)   -> (c, grad_x)
%   cg.f_V_os (x, y, d_V, h, side)           -> (c, grad_x)  (单边)
%   cg.f_BC_os(x, y, d_BC, gamma, a, b, h, side) -> (c, grad_x)
%   cg.f_obj  (x, y_obj, h, tau_obj)         -> (val, grad_x)  (y_obj=[nu;mu])

import casadi.*

%% ---- 从 params 解包 ----
nb = params.num_bus;
nL = params.num_load;
nD = params.num_DG;
MM = params.num_samples;
load_buses = params.load_buses;
DG_buses   = params.DG_buses;
V0 = params.V0;
V_max = params.V_max; V_min = params.V_min;
Vc = (V_max + V_min) / 2;
eps_bi = params.epsilon;           % 双边用 epsilon
eps_os = params.epsilon / 2;       % 单边 Bonferroni, 用 eps/2
tau_cc = params.tau;
id = params.idx;
eta1 = params.eta1; eta2 = params.eta2;
xi_num = params.xi;                 % (nL+nD) x MM

%% ---- 符号决策变量 x (与主问题一致) ----
x = SX.sym('x', id.nx, 1);
pc_s  = x(id.pcut);
qc_s  = x(id.qcut);
lam_s = x(id.lambda);

% 净注入 pn(1:nb), qn(1:nb) (符号)
pn_s = SX.zeros(nb, 1);
qn_s = SX.zeros(nb, 1);
for i = 1:nL
    b = load_buses(i);
    pn_s(b) = pn_s(b) + params.P_L(b) - pc_s(i);
    qn_s(b) = qn_s(b) + params.Q_L(b) - qc_s(i);
end
for i = 1:nD
    b = DG_buses(i);
    pn_s(b) = pn_s(b) - params.P_DG(i) * (1 - lam_s(i));
end
pn_tail = pn_s(2:end);   % nb-1
qn_tail = qn_s(2:end);

%% ---- 子问题对偶变量 (对约束) ----
y = SX.sym('y', 3, 1);
eta_s = y(1); nu_s = y(2); mu_s = y(3);

%% ====================================================================
%% (1) 双边电压约束
%% ====================================================================
% 数据向量 d_V = [aL(nL); aDG(nD); R_row(nb-1); X_row(nb-1)]
dV_dim = nL + nD + 2*(nb-1);
d_V = SX.sym('d_V', dV_dim, 1);
aL_V  = d_V(1:nL);
aDG_V = d_V(nL+1 : nL+nD);
Rrow  = d_V(nL+nD+1 : nL+nD+nb-1);
Xrow  = d_V(nL+nD+nb : nL+nD+2*(nb-1));
h_V_sym = SX.sym('h_V');

% V_bar(x) = V0 - pn_tail'*R_row - qn_tail'*X_row
V_bar = V0 - dot(pn_tail, Rrow) - dot(qn_tail, Xrow);
% 对双边电压: gamma = gV = (V_max-V_min)/2, beta = V_bar - Vc
gV_const = (V_max - V_min) / 2;
beta_V_bi = V_bar - Vc;
aDG_eff_V = aDG_V .* (1 - lam_s);

c_V_bi_expr = bilateral_c(eta_s, nu_s, mu_s, ...
    aL_V, aDG_eff_V, beta_V_bi, gV_const, h_V_sym, ...
    xi_num(1:nL,:), xi_num(nL+1:end,:), eps_bi, tau_cc, MM);
grad_V_bi = jacobian(c_V_bi_expr, x)';

cg.f_V_bi = Function('f_V_bi', ...
    {x, y, d_V, h_V_sym}, {c_V_bi_expr, grad_V_bi}, ...
    {'x','y','d_V','h'}, {'c','grad'});

%% ====================================================================
%% (2) 双边支路约束
%% ====================================================================
% 数据: d_BC = [aL(nL); aDG(nD); mask_ds(nb-1)] + 标量 gamma, a, b, h
dBC_dim = nL + nD + (nb-1);
d_BC = SX.sym('d_BC', dBC_dim, 1);
aL_BC   = d_BC(1:nL);
aDG_BC  = d_BC(nL+1 : nL+nD);
mask_BC = d_BC(nL+nD+1 : nL+nD+nb-1);
gamma_BC_s = SX.sym('gamma_BC');
a_coef = SX.sym('a_coef');
b_coef = SX.sym('b_coef');
h_BC_sym = SX.sym('h_BC');

% beta_BC = a*pn_ds + b*qn_ds, pn_ds = sum_{j in ds, j>=2} pn(j) = mask' * pn_tail
beta_BC = a_coef * dot(pn_tail, mask_BC) + b_coef * dot(qn_tail, mask_BC);
aDG_eff_BC = aDG_BC .* (1 - lam_s);

c_BC_bi_expr = bilateral_c(eta_s, nu_s, mu_s, ...
    aL_BC, aDG_eff_BC, beta_BC, gamma_BC_s, h_BC_sym, ...
    xi_num(1:nL,:), xi_num(nL+1:end,:), eps_bi, tau_cc, MM);
grad_BC_bi = jacobian(c_BC_bi_expr, x)';

cg.f_BC_bi = Function('f_BC_bi', ...
    {x, y, d_BC, gamma_BC_s, a_coef, b_coef, h_BC_sym}, ...
    {c_BC_bi_expr, grad_BC_bi}, ...
    {'x','y','d_BC','gamma','a','b','h'}, {'c','grad'});

%% ====================================================================
%% (3) 单边电压约束 (side=+1 上界, side=-1 下界)
%% 对单边, 使用 Y_full = V_bar + aL'*xi_L + aDG_eff'*xi_DG (不减 Vc)
%%   上界约束: P(Y_full > V_max) <= eps/2, 对应 side=+1, gamma_eff = V_max
%%   下界约束: P(-Y_full > -V_min) <= eps/2, 对应 side=-1, gamma_eff = -V_min
%% ====================================================================
side_s = SX.sym('side');
h_V_sym_os = SX.sym('h_V_os');

% 单边用 beta_full = V_bar (不减 Vc), gamma 由调用方传入
% 为与双边共用 d_V 结构 (不含 gamma), 这里单独定义 gamma_V_os
gamma_V_os = SX.sym('gamma_V_os');
beta_V_os = V_bar;  % 完整 V_bar
c_V_os_expr = oneside_c(eta_s, nu_s, mu_s, ...
    aL_V, aDG_eff_V, beta_V_os, gamma_V_os, h_V_sym_os, side_s, ...
    xi_num(1:nL,:), xi_num(nL+1:end,:), eps_os, tau_cc, MM);
grad_V_os = jacobian(c_V_os_expr, x)';

cg.f_V_os = Function('f_V_os', ...
    {x, y, d_V, gamma_V_os, h_V_sym_os, side_s}, ...
    {c_V_os_expr, grad_V_os}, ...
    {'x','y','d_V','gamma','h','side'}, {'c','grad'});

%% ====================================================================
%% (4) 单边支路约束
%% ====================================================================
% 支路: beta_BC 与双边相同; 上界 side=+1, gamma=gam; 下界 side=-1, gamma=gam
gamma_BC_os = SX.sym('gamma_BC_os');
a_coef_os = SX.sym('a_coef_os');
b_coef_os = SX.sym('b_coef_os');
h_BC_sym_os = SX.sym('h_BC_os');
beta_BC_os = a_coef_os * dot(pn_tail, mask_BC) + b_coef_os * dot(qn_tail, mask_BC);

c_BC_os_expr = oneside_c(eta_s, nu_s, mu_s, ...
    aL_BC, aDG_eff_BC, beta_BC_os, gamma_BC_os, h_BC_sym_os, side_s, ...
    xi_num(1:nL,:), xi_num(nL+1:end,:), eps_os, tau_cc, MM);
grad_BC_os = jacobian(c_BC_os_expr, x)';

cg.f_BC_os = Function('f_BC_os', ...
    {x, y, d_BC, gamma_BC_os, a_coef_os, b_coef_os, h_BC_sym_os, side_s}, ...
    {c_BC_os_expr, grad_BC_os}, ...
    {'x','y','d_BC','gamma','a','b','h','side'}, {'c','grad'});

%% ====================================================================
%% (5) DRO 目标子问题
%%   g1(x, xi_m) = sum_i xi_L(i,m) - sum_i (1-lam_i)*xi_DG(i,m)
%%   d_m = (eta1-eta2)/2 * [Psi(-g1m,h) + Psi(g1m,h)] + (eta1+eta2)/2 * g1m
%%   子问题 (消去 v_m, 无 eta): 2维 (nu, mu)
%%     val = nu(tau-1) + mu + 1/(4M nu) * sum ([d_m - mu + 2nu]_+)^2
%% ====================================================================
y_obj = SX.sym('y_obj', 2, 1);
nu_o = y_obj(1); mu_o = y_obj(2);
h_obj_sym = SX.sym('h_obj');
tau_obj_sym = SX.sym('tau_obj');

g1_vec = SX.zeros(1, MM);
for m = 1:MM
    v_sum = 0;
    for i = 1:nL
        v_sum = v_sum + xi_num(i, m);
    end
    for i = 1:nD
        v_sum = v_sum - (1 - lam_s(i)) * xi_num(nL+i, m);
    end
    g1_vec(m) = v_sum;
end

psi_obj_sum = 0;
for m = 1:MM
    g1m = g1_vec(m);
    d_m_obj = (eta1-eta2)/2 * (casadi_Psi(-g1m, h_obj_sym) + casadi_Psi(g1m, h_obj_sym)) ...
            + (eta1+eta2)/2 * g1m;
    t_m = d_m_obj - mu_o + 2*nu_o;
    tp_m = fmax(t_m, 0);
    psi_obj_sum = psi_obj_sum + tp_m^2;
end
f_obj_expr = nu_o*(tau_obj_sym - 1) + mu_o + psi_obj_sum / (4*MM*fmax(nu_o, 1e-10));
grad_obj = jacobian(f_obj_expr, x)';

cg.f_obj = Function('f_obj', ...
    {x, y_obj, h_obj_sym, tau_obj_sym}, ...
    {f_obj_expr, grad_obj}, ...
    {'x','y','h','tau'}, {'val','grad'});

end

%% ============================================================
%%                    符号辅助子程序
%% ============================================================
function Psi_out = casadi_Psi(u, h)
    import casadi.*
    h_safe = fmax(h, 1e-12);
    z = u / h_safe;
    z = fmax(z, -30);
    z = fmin(z, 30);
    Phi_z = 0.5 * (1 + erf(z / sqrt(2)));
    phi_z = exp(-0.5 * z^2) / sqrt(2 * pi);
    Psi_kde = u * Phi_z + h_safe * phi_z;
    Psi_h0 = fmax(u, 0);
    Psi_out = if_else(h > 1e-10, Psi_kde, Psi_h0);
end

% 双边子问题目标 c(x,y):
%   c = eta + nu(tau-1) + mu + 1/(4M nu) * sum_m ([d_m/eps - mu + 2nu]_+)^2
%   d_m = Psi(-Y_m - C+, h) + Psi(Y_m - C+, h) + C-,  C = gamma + eta
%   Y_m = aL' * xi_L(:,m) + aDG_eff' * xi_DG(:,m) + beta
function c_expr = bilateral_c(eta_v, nu_v, mu_v, aL, aDG_eff, beta_v, ...
        gamma_v, h_v, xi_L, xi_DG, eps_v, tau_v, MM)
    import casadi.*
    C_v = gamma_v + eta_v;
    Cp = if_else(C_v >= 0, C_v, 0);
    Cm = if_else(C_v >= 0, 0, -C_v);
    ps = 0;
    for m = 1:MM
        Y_m = dot(aL, xi_L(:,m)) + dot(aDG_eff, xi_DG(:,m)) + beta_v;
        u1_m = -Y_m - Cp;
        u2_m =  Y_m - Cp;
        d_m = (casadi_Psi(u1_m, h_v) + casadi_Psi(u2_m, h_v) + Cm) / eps_v;
        t_m = d_m - mu_v + 2*nu_v;
        tp_m = fmax(t_m, 0);
        ps = ps + tp_m^2;
    end
    c_expr = eta_v + nu_v*(tau_v - 1) + mu_v + ps / (4*MM*fmax(nu_v, 1e-10));
end

% 单边子问题目标 c(x,y):
%   c = eta + nu(tau-1) + mu + 1/(4M nu) * sum_m ([d_m/eps_os - mu + 2nu]_+)^2
%   d_m = Psi(side * Y_m - gamma - eta, h)
%   Y_m = aL'*xi_L + aDG_eff'*xi_DG + beta  (这里 beta 是"完整" Y_full)
function c_expr = oneside_c(eta_v, nu_v, mu_v, aL, aDG_eff, beta_v, ...
        gamma_v, h_v, side, xi_L, xi_DG, eps_v, tau_v, MM)
    import casadi.*
    ps = 0;
    for m = 1:MM
        Y_m = dot(aL, xi_L(:,m)) + dot(aDG_eff, xi_DG(:,m)) + beta_v;
        arg_m = side * Y_m - gamma_v - eta_v;
        d_m = casadi_Psi(arg_m, h_v) / eps_v;
        t_m = d_m - mu_v + 2*nu_v;
        tp_m = fmax(t_m, 0);
        ps = ps + tp_m^2;
    end
    c_expr = eta_v + nu_v*(tau_v - 1) + mu_v + ps / (4*MM*fmax(nu_v, 1e-10));
end