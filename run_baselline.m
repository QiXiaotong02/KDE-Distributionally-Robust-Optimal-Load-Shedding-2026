%% run_baseline.m
% 基准实验: 24h 四模型对比 (M1/M2/M3-单边/M4-双边)
% =========================================================================
clear; clc; close all;
addpath('casadi path');

fprintf('================================================================\n');
fprintf('  基准实验: 24h 四模型对比 (M1/M2/M3/M4)\n');
fprintf('================================================================\n');

%% 加载 CasADi
try
    import casadi.*
    casadi.SX.sym('t_chk');
    fprintf('  CasADi 已加载\n');
catch
    error(['CasADi 未安装或未添加到路径. 请:\n' ...
           '  addpath(''你的 casadi-matlabR2014a-v3.5.5 路径'')\n']);
end

%% 加载数据
if ~exist('init_data.mat','file')
    fprintf('  未找到 init_data.mat, 正在运行 init_data.m ...\n');
    run('init_data.m');
end
load('init_data.mat');

sf = shared_functions();

%% 构建 CasADi 梯度函数
fprintf('\n  逐时段求解中...\n');

n_models = 4;
x_all = cell(n_models, T_periods);
cost_all = zeros(n_models, T_periods);
V_all = cell(n_models, T_periods);
Sbr_all = cell(n_models, T_periods);
niter_all = zeros(n_models, T_periods);
time_all = zeros(n_models, 1);
time_per_t = zeros(n_models, T_periods);

model_names = {'M1:Deterministic','M2:Moment-DRCCO','M3:KDE-Oneside','M4:KDE-Bilateral'};

for t = 1:T_periods
    fprintf('  t=%2d/%d ...', t, T_periods);

    pt = sf.build_params(t, P_load_t, Q_load_t, P_DG_forecast, xi_all{t}, ...
        num_bus, num_branch, num_load, num_DG, num_samples, num_segments, ...
        load_buses, DG_buses, from_bus, to_bus, W, S_max_br, ...
        a_s, b_s, c_s, V0, V_min, V_max, tan_theta, ...
        epsilon, tau, tau_obj, eta1, eta2, eta3, baseMVA, ...
        R_mat, X_mat, A_V_load_base, A_V_DG_base, ...
        A_BC_load_base, A_BC_DG_base, A_obj_load_base, A_obj_DG_base, ...
        BC_info, n_V_constr, n_BC_constr, idx, Sigma_all{t}, DG_types);

    % 为本时段构建 CasADi 梯度
    cg = build_casadi_grads_wrapper(pt);

    lb_t = zeros(nx_per_t, 1);
    ub_t = inf(nx_per_t, 1);
    for i = 1:num_load
        ub_t(idx.pcut(i)) = pt.P_L(load_buses(i));
        ub_t(idx.qcut(i)) = pt.Q_L(load_buses(i));
    end
    ub_t(idx.lambda) = 1;
    ub_t(idx.pplus) = 10;
    ub_t(idx.pminus) = 10;

    x0t = zeros(nx_per_t, 1);
    nl = sum(pt.P_L(2:end)) - sum(pt.P_DG);
    x0t(idx.pplus) = max(0, nl);
    x0t(idx.pminus) = max(0, -nl);

    opts_f = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
        'MaxIterations', 500, 'OptimalityTolerance', 1e-6, 'ConstraintTolerance', 1e-6);

    % ---- M1 ----
    tic_m = tic;
    try
        [x_d, f_d] = fmincon(@(xx) sf.obj_det(xx, pt), x0t, [], [], [], [], ...
            lb_t, ub_t, @(xx) sf.con_det(xx, pt), opts_f);
    catch
        x_d = x0t; f_d = sf.obj_det(x0t, pt);
    end
    t_m1 = toc(tic_m);
    time_all(1) = time_all(1) + t_m1; time_per_t(1,t) = t_m1;
    [V_d, ~, ~, S_d] = sf.calc_pf(x_d, pt);
    x_all{1,t} = x_d; cost_all(1,t) = f_d;
    V_all{1,t} = V_d; Sbr_all{1,t} = S_d;

    % ---- M2 ----
    tic_m = tic;
    [x_h, f_h, ~] = sf.solve_moment_DRCCO_t(pt);
    t_m2 = toc(tic_m);
    time_all(2) = time_all(2) + t_m2; time_per_t(2,t) = t_m2;
    [V_h, ~, ~, S_h] = sf.calc_pf(x_h, pt);
    x_all{2,t} = x_h; cost_all(2,t) = f_h;
    V_all{2,t} = V_h; Sbr_all{2,t} = S_h;

    % ---- M3 ----
    pt3 = pt; pt3.tau = tau; pt3.tau_obj = tau_obj; pt3.use_kde = true;
    tic_m = tic;
    [x_k3, f_k3, ni_k3] = sf.solve_cp_oneside(x_d, pt3, lb_t, ub_t, opts_f, cg);
    t_m3 = toc(tic_m);
    time_all(3) = time_all(3) + t_m3; time_per_t(3,t) = t_m3;
    [V_k3, ~, ~, S_k3] = sf.calc_pf(x_k3, pt);
    x_all{3,t} = x_k3; cost_all(3,t) = f_k3; niter_all(3,t) = ni_k3;
    V_all{3,t} = V_k3; Sbr_all{3,t} = S_k3;

    % ---- M4 ----
    pt4 = pt; pt4.tau = tau; pt4.tau_obj = tau_obj; pt4.use_kde = true;
    tic_m = tic;
    [x_k4, f_k4, ni_k4] = sf.solve_cp_analytical(x_d, pt4, lb_t, ub_t, opts_f, cg);
    t_m4 = toc(tic_m);
    time_all(4) = time_all(4) + t_m4; time_per_t(4,t) = t_m4;
    [V_k4, ~, ~, S_k4] = sf.calc_pf(x_k4, pt);
    x_all{4,t} = x_k4; cost_all(4,t) = f_k4; niter_all(4,t) = ni_k4;
    V_all{4,t} = V_k4; Sbr_all{4,t} = S_k4;

    fprintf(' M1=$%.2f M2=$%.2f M3=$%.2f(i%d) M4=$%.2f(i%d)\n', ...
        f_d*baseMVA, f_h*baseMVA, f_k3*baseMVA, ni_k3, f_k4*baseMVA, ni_k4);
end

%% 蒙特卡洛验证
fprintf('\n  蒙特卡洛验证 (5000次/时段)...\n');
n_mc = 5000; rng(123);
mc_V = zeros(n_models, T_periods);
mc_BC = zeros(n_models, T_periods);
mc_cost = zeros(n_models, T_periods);

for t = 1:T_periods
    PL_t = P_load_t(:, t); QL_t = Q_load_t(:, t); PDG_t = P_DG_forecast(:, t);
    for mdl = 1:n_models
        xm = x_all{mdl, t};
        lam = xm(idx.lambda);
        vd=0; bd=0; cd=0;
        for mc_i = 1:n_mc
            xi_mc = zeros(num_load + num_DG, 1);
            for i = 1:num_load
                xi_mc(i) = sigma_std * PL_t(load_buses(i)) * randn();
            end
            for i = 1:num_DG
                xi_mc(num_load+i) = sigma_std * PDG_t(i) * randn();
            end
            pn=zeros(num_bus,1); qn=zeros(num_bus,1);
            for i=1:num_load
                bus=load_buses(i);
                pn(bus)=PL_t(bus)-xm(idx.pcut(i))+xi_mc(i);
                qn(bus)=QL_t(bus)-xm(idx.qcut(i))+xi_mc(i)*tan_theta;
            end
            for i=1:num_DG
                bus=DG_buses(i);
                pn(bus)=pn(bus)-PDG_t(i)*(1-lam(i))+(1-lam(i))*xi_mc(num_load+i);
            end
            vv=false;
            for i=2:num_bus
                Vi=V0-sum(pn(2:end).*R_mat(i,2:end)'+qn(2:end).*X_mat(i,2:end)');
                if Vi<V_min-1e-6||Vi>V_max+1e-6, vv=true; break; end
            end
            bb=false;
            for br=1:num_branch
                ds=W{to_bus(br)};
                if sqrt(sum(pn(ds))^2+sum(qn(ds))^2)>S_max_br(br)+1e-6, bb=true; break; end
            end
            dp=sum(xi_mc(1:num_load))-sum((1-lam).*xi_mc(num_load+1:end));
            cc=eta1*max(xm(idx.pplus)+dp,0)-eta2*max(xm(idx.pminus)-dp,0)+eta3*sum(xm(idx.pcut));
            vd=vd+vv; bd=bd+bb; cd=cd+cc;
        end
        mc_V(mdl,t)=100*vd/n_mc;
        mc_BC(mdl,t)=100*bd/n_mc;
        mc_cost(mdl,t)=cd/n_mc;
    end
end

%% 汇总
fprintf('\n================================================================\n');
fprintf('                    24h 结果汇总                                ');
fprintf('================================================================\n');

tc = zeros(n_models,1); ts = zeros(n_models,1); tcu = zeros(n_models,1);
tbuy = zeros(n_models,1); tsell = zeros(n_models,1);
for mdl=1:n_models
    for t=1:T_periods
        tc(mdl) = tc(mdl) + cost_all(mdl,t)*baseMVA*Delta_t;
        ts(mdl) = ts(mdl) + sum(x_all{mdl,t}(idx.pcut))*baseMVA;
        tcu(mdl) = tcu(mdl) + sum(x_all{mdl,t}(idx.lambda).*P_DG_forecast(:,t))*baseMVA;
        tbuy(mdl) = tbuy(mdl) + x_all{mdl,t}(idx.pplus)*baseMVA;
        tsell(mdl) = tsell(mdl) + x_all{mdl,t}(idx.pminus)*baseMVA;
    end
end

glo_Vmin = zeros(n_models,1); glo_Vmax = zeros(n_models,1); glo_BRmax = zeros(n_models,1);
for mdl=1:n_models
    vmin_v = inf; vmax_v = -inf; brmax_v = 0;
    for t=1:T_periods
        vmin_v = min(vmin_v, min(V_all{mdl,t}(2:end)));
        vmax_v = max(vmax_v, max(V_all{mdl,t}(2:end)));
        brmax_v = max(brmax_v, 100*max(Sbr_all{mdl,t}./S_max_br));
    end
    glo_Vmin(mdl) = vmin_v;
    glo_Vmax(mdl) = vmax_v;
    glo_BRmax(mdl) = brmax_v;
end

mc_total_cost = zeros(n_models,1);
for mdl=1:n_models
    mc_total_cost(mdl) = sum(mc_cost(mdl,:)) * baseMVA * Delta_t;
end

fprintf('\n                  M1:Det       M2:Moment    M3:KDE-1side M4:KDE-2side\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('24h优化成本($):  %10.4f %10.4f %10.4f %10.4f\n', tc(1),tc(2),tc(3),tc(4));
fprintf('24hMC总成本($):  %10.4f %10.4f %10.4f %10.4f\n', mc_total_cost(1),mc_total_cost(2),mc_total_cost(3),mc_total_cost(4));
fprintf('24h总切负荷(MWh):%10.4f %10.4f %10.4f %10.4f\n', ts(1),ts(2),ts(3),ts(4));
fprintf('24h总DG削减(MWh):%10.4f %10.4f %10.4f %10.4f\n', tcu(1),tcu(2),tcu(3),tcu(4));
fprintf('全局最低电压(pu): %10.4f %10.4f %10.4f %10.4f\n', glo_Vmin(1),glo_Vmin(2),glo_Vmin(3),glo_Vmin(4));
fprintf('全局最高电压(pu): %10.4f %10.4f %10.4f %10.4f\n', glo_Vmax(1),glo_Vmax(2),glo_Vmax(3),glo_Vmax(4));
fprintf('最高支路负载(%%):  %10.2f %10.2f %10.2f %10.2f\n', glo_BRmax(1),glo_BRmax(2),glo_BRmax(3),glo_BRmax(4));
fprintf('平均V违反(%%):    %10.2f %10.2f %10.2f %10.2f  (<%.0f%%)\n',...
    mean(mc_V(1,:)),mean(mc_V(2,:)),mean(mc_V(3,:)),mean(mc_V(4,:)),100*epsilon);
fprintf('最大V违反(%%):    %10.2f %10.2f %10.2f %10.2f\n',...
    max(mc_V(1,:)),max(mc_V(2,:)),max(mc_V(3,:)),max(mc_V(4,:)));
fprintf('平均Br违反(%%):   %10.2f %10.2f %10.2f %10.2f  (<%.0f%%)\n',...
    mean(mc_BC(1,:)),mean(mc_BC(2,:)),mean(mc_BC(3,:)),mean(mc_BC(4,:)),100*epsilon);
rel = zeros(n_models,1);
for mdl=1:n_models
    rel(mdl) = 100-mean(max(mc_V(mdl,:),mc_BC(mdl,:)));
end
fprintf('可靠性(%%):       %10.2f %10.2f %10.2f %10.2f\n', rel(1),rel(2),rel(3),rel(4));
fprintf('求解时间(s):     %10.2f %10.2f %10.2f %10.2f\n', time_all(1),time_all(2),time_all(3),time_all(4));
fprintf('------------------------------------------------------------------------\n');

%% 保存
save_fn = 'baseline_results.mat';
save(save_fn, 'x_all', 'cost_all', 'V_all', 'Sbr_all', ...
    'niter_all', 'time_all', 'time_per_t', 'mc_V', 'mc_BC', 'mc_cost', ...
    'tc', 'ts', 'tcu', 'tbuy', 'tsell', 'glo_Vmin', 'glo_Vmax', ...
    'glo_BRmax', 'mc_total_cost', 'model_names');
fprintf('\n  结果保存到 %s\n', save_fn);
%% 绘图
hours = 1:T_periods;
clrs = [0.2 0.4 0.8; 0.2 0.7 0.3; 0.9 0.5 0.0; 0.8 0.2 0.2];
lw = 1.5;

figure('Name','24h Cost');
bar(hours, [cost_all(1,:)'*baseMVA, cost_all(2,:)'*baseMVA, ...
    cost_all(3,:)'*baseMVA, cost_all(4,:)'*baseMVA]);
xlabel('Hour'); ylabel('Cost ($)');
legend(model_names,'Location','best');
title(sprintf('Hourly Cost ')); grid on;

figure('Name','Reliability');
subplot(1,2,1); bar(hours, mc_V'); hold on;
plot([0.5 24.5],[100*epsilon 100*epsilon],'r--','LineWidth',1.5);
xlabel('Hour'); ylabel('V Viol (%)'); legend(model_names,'Location','best'); grid on;
title('Voltage Violation');
subplot(1,2,2); bar(hours, mc_BC'); hold on;
plot([0.5 24.5],[100*epsilon 100*epsilon],'r--','LineWidth',1.5);
xlabel('Hour'); ylabel('Br Viol (%)'); legend(model_names,'Location','best'); grid on;
title('Branch Violation');

fprintf('\n  基准实验完成!\n');
fprintf('================================================================\n');

%% ============== 构建 CasADi 梯度 ==============
function cg = build_casadi_grads_wrapper(pt)
    cg = build_casadi_grads(pt);
end
