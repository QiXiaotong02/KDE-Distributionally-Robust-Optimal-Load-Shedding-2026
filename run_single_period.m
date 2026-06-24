%% run_single_period.m
% 单时段高负荷场景 (M1 / M2 / M3 / M4 四模型对比)
% =========================================================================

clear; clc; close all;
addpath('D:/C盘来的/casadi-windows-matlabR2016a-v3.5.5');

fprintf('================================================================\n');
fprintf('  单时段场景: 四模型对比 (M1 / M2 / M3 / M4) \n');
fprintf('================================================================\n');

%% CasADi
try
    import casadi.*
    casadi.SX.sym('t_chk');
    fprintf('  CasADi 已加载\n');
catch
    error('CasADi 未加载, 请 addpath 到 CasADi 根目录');
end

sf = shared_functions();

%% ==================== 1. 系统数据 ====================
fprintf('\n[1] 读取系统数据...\n');
% mpc = loadcase('case33bw');
mpc = loadcase('grid_IEEE123');
baseMVA = mpc.baseMVA;
num_bus = size(mpc.bus, 1);
num_branch = size(mpc.branch, 1);
bus_data = mpc.bus; branch_data = mpc.branch;
from_bus = branch_data(:, 1);
to_bus = branch_data(:, 2);
r = branch_data(:, 3); x = branch_data(:, 4);
rateA = branch_data(:, 6);
rateA_pu = rateA / baseMVA;

V0 = 1.0; V_max = 1.05; V_min = 0.95;
power_factor = 0.85;
tan_theta = tan(acos(power_factor));

eta1 = 90; eta2 = 45; eta3 = 300;
epsilon = 0.05; tau = 0.01; tau_obj = 0.01;
num_samples = 100; num_segments = 12;
MM = num_samples;

a_s = [1,1,0.2679,-0.2679,-1,-1,-1,-1,-0.2679,0.2679,1,1];
b_s = [0.2679,1,1,1,1,0.2679,-0.2679,-1,-1,-1,-1,-0.2679];
c_s = -[-1,-1.366,-1,-1,-1.366,-1,-1,-1.366,-1,-1,-1.366,-1];

fprintf('  节点=%d, 支路=%d, 样本=%d, tau=%.4f, eps=%.3f\n', ...
    num_bus, num_branch, num_samples, tau, epsilon);

%% 1.2 负荷 & DG
load_scale = 1.3;
P_load = load_scale * max(0, bus_data(:,3)/baseMVA);
Q_load = load_scale * max(0, bus_data(:,4)/baseMVA);

% --- IEEE 123 ---
DG_buses = [30, 60, 80, 100, 114];
DG_types = {'pv','wind','pv','wind','pv'};
num_DG = length(DG_buses);
P_DG_forecast = zeros(num_DG, 1);
for i = 1:num_DG
    P_DG_forecast(i) = 0.1 * mpc.bus(DG_buses(i), 3) / mpc.baseMVA;
end
% %--- IEEE 33  ---
% DG_buses = [18, 22, 25, 33];
% DG_types = {'wind','pv','wind','pv'};
% num_DG = length(DG_buses);
% P_DG_forecast = zeros(num_DG, 1);
% for i = 1:num_DG
%     P_DG_forecast(i) = 0.5 * P_load(DG_buses(i)) * 0.2;
% end

load_buses = find(P_load > 1e-6);
load_buses = load_buses(load_buses ~= 1);
num_load = length(load_buses);
total_load = sum(P_load);

S_max_br = rateA_pu;
S_max_br(S_max_br < 1e-6) = 2 * total_load;

%% 1.3 网络矩阵
fprintf('\n[2] 计算网络矩阵...\n');
R_mat = zeros(num_bus); X_mat = zeros(num_bus);
for i = 2:num_bus
    pi_ = local_path_to_root(i, from_bus, to_bus);
    for j = 2:num_bus
        pj = local_path_to_root(j, from_bus, to_bus);
        cm = intersect(pi_, pj);
        R_mat(i,j) = sum(r(cm)); X_mat(i,j) = sum(x(cm));
    end
end
W = cell(num_bus, 1);
for j = 1:num_bus, W{j} = local_downstream(j, from_bus, to_bus); end

%% 1.4 灵敏度
fprintf('\n[3] 灵敏度矩阵...\n');
n_V_constr = num_bus - 1;
n_BC_constr = num_branch * num_segments / 2;

A_V_load_base = zeros(n_V_constr, num_load);
A_V_DG_base = zeros(n_V_constr, num_DG);
for k = 1:n_V_constr
    ib = k+1;
    for jj = 1:num_load
        A_V_load_base(k,jj) = -(R_mat(ib,load_buses(jj)) + X_mat(ib,load_buses(jj))*tan_theta);
    end
    for jj = 1:num_DG
        A_V_DG_base(k,jj) = R_mat(ib, DG_buses(jj));
    end
end

A_BC_load_base = zeros(n_BC_constr, num_load);
A_BC_DG_base = zeros(n_BC_constr, num_DG);
BC_info = zeros(n_BC_constr, 2);
ci = 0;
for br = 1:num_branch
    ds = W{to_bus(br)};
    for s = 1:num_segments/2
        ci = ci+1; BC_info(ci,:) = [br, s];
        for kk = 1:num_load
            if ismember(load_buses(kk), ds)
                A_BC_load_base(ci,kk) = a_s(s) + b_s(s)*tan_theta;
            end
        end
        for kk = 1:num_DG
            if ismember(DG_buses(kk), ds)
                A_BC_DG_base(ci,kk) = -a_s(s);
            end
        end
    end
end

A_obj_load_base = ones(1, num_load);
A_obj_DG_base = -ones(1, num_DG);

%% 1.5 样本
fprintf('\n[4] 生成样本 (sigma=0.15)...\n');
rng(42);
sigma_load = 0.15; sigma_DG = 0.15;
xi = zeros(num_load + num_DG, num_samples);
for i = 1:num_load
    xi(i,:) = sigma_load * P_load(load_buses(i)) * randn(1, num_samples);
end
for i = 1:num_DG
    xi(num_load+i,:) = sigma_DG * P_DG_forecast(i) * randn(1, num_samples);
end
Sigma_diag = zeros(num_load + num_DG, 1);
for i = 1:num_load, Sigma_diag(i) = (sigma_load*P_load(load_buses(i)))^2; end
for i = 1:num_DG, Sigma_diag(num_load+i) = (sigma_DG*P_DG_forecast(i))^2; end

%% 1.6 索引
n_pcut = num_load; n_qcut = num_load; n_lambda = num_DG;
nx = n_pcut + n_qcut + n_lambda + 2;
idx.pcut   = 1:n_pcut;
idx.qcut   = n_pcut + (1:n_qcut);
idx.lambda = n_pcut + n_qcut + (1:n_lambda);
idx.pplus  = n_pcut + n_qcut + n_lambda + 1;
idx.pminus = n_pcut + n_qcut + n_lambda + 2;
idx.nx = nx;

%% ==== 构造 params ====
P_load_t = P_load;
Q_load_t = Q_load;
P_DG_f   = P_DG_forecast;

params = sf.build_params(1, P_load_t, Q_load_t, P_DG_f, xi, ...
    num_bus, num_branch, num_load, num_DG, num_samples, num_segments, ...
    load_buses, DG_buses, from_bus, to_bus, W, S_max_br, ...
    a_s, b_s, c_s, V0, V_min, V_max, tan_theta, ...
    epsilon, tau, tau_obj, eta1, eta2, eta3, baseMVA, ...
    R_mat, X_mat, A_V_load_base, A_V_DG_base, ...
    A_BC_load_base, A_BC_DG_base, A_obj_load_base, A_obj_DG_base, ...
    BC_info, n_V_constr, n_BC_constr, idx, Sigma_diag, DG_types);

%% ==== 构建 CasADi 梯度 ====
fprintf('\n[5] 构建 CasADi 自动微分...\n');
tic_cg = tic;
cg = build_casadi_grads(params);
fprintf('  CasADi 图构建耗时: %.2f s\n', toc(tic_cg));

%% ==================== 2. 边界与初值 ====================
lb_all = zeros(nx,1); ub_all = inf(nx,1);
for i = 1:num_load
    ub_all(idx.pcut(i)) = P_load(load_buses(i));
    ub_all(idx.qcut(i)) = Q_load(load_buses(i));
end
ub_all(idx.lambda) = 1;
ub_all(idx.pplus) = 10; ub_all(idx.pminus) = 10;

x0 = zeros(nx, 1);
nl = sum(P_load(2:end)) - sum(P_DG_forecast);
x0(idx.pplus) = max(0, nl);
x0(idx.pminus) = max(0, -nl);

% 主问题
opts_f = optimoptions('fmincon','Display','off','Algorithm','sqp',...
    'MaxIterations',500,'MaxFunctionEvaluations',5e4,...
    'OptimalityTolerance',1e-6,'ConstraintTolerance',1e-6);
% 备用算法
opts_ip = optimoptions('fmincon','Display','off','Algorithm','interior-point',...
    'MaxIterations',500,'OptimalityTolerance',1e-6,'ConstraintTolerance',1e-6);

%% ==================== 3. 求解四个模型 ====================
% ---------- M1: 确定性 ----------
fprintf('\n[6.1] M1 (确定性) 求解...\n');
t_m1_start = tic;
try
    [x_m1, fval_m1] = fmincon(@(xx) sf.obj_det(xx, params), x0, [],[], [],[], ...
        lb_all, ub_all, @(xx) sf.con_det(xx, params), opts_f);
catch
    x_m1 = x0; fval_m1 = sf.obj_det(x0, params);
end
t_m1 = toc(t_m1_start);
fprintf('  M1 cost=$%.4f, shed=%.4f MW, time=%.2fs\n', ...
    fval_m1*baseMVA, sum(x_m1(idx.pcut))*baseMVA, t_m1);

% ---------- M2: Moment-DRCCO ----------
fprintf('\n[6.2] M2 (Moment-DRCCO) 求解...\n');
t_m2_start = tic;
try
    [x_m2, fval_m2, ~] = sf.solve_moment_DRCCO_t(params);
catch ME
    fprintf('  M2 失败: %s, 用 M1 代替\n', ME.message);
    x_m2 = x_m1; fval_m2 = fval_m1;
end
t_m2 = toc(t_m2_start);
fprintf('  M2 cost=$%.4f, shed=%.4f MW, time=%.2fs\n', ...
    fval_m2*baseMVA, sum(x_m2(idx.pcut))*baseMVA, t_m2);

% ---------- M3: KDE-DRCCO 单边 ----------
fprintf('\n================================================================\n');
fprintf('[6.3] M3 (KDE-DRCCO 单边 / Bonferroni) 割平面求解\n');
fprintf('================================================================\n');

params3 = params;
params3.use_kde = true;

t_m3_start = tic;
[x_opt3, fval_m3, ni3, history3] = run_cutplane( ...
    x_m1, params3, lb_all, ub_all, opts_f, opts_ip, cg, sf, 'oneside');
t_m3 = toc(t_m3_start);

fprintf('  M3 final cost=$%.4f, shed=%.4f MW, iters=%d, time=%.2fs\n', ...
    fval_m3*baseMVA, sum(x_opt3(idx.pcut))*baseMVA, ni3, t_m3);

% ---------- M4: KDE-DRCCO 双边 ----------
fprintf('\n================================================================\n');
fprintf('[6.4] M4 (KDE-DRCCO 双边 / 本文方法) 割平面求解\n');
fprintf('================================================================\n');

params4 = params;
params4.use_kde = true;

t_m4_start = tic;
[x_opt4, fval_m4, ni4, history4] = run_cutplane( ...
    x_m1, params4, lb_all, ub_all, opts_f, opts_ip, cg, sf, 'bilateral');
t_m4 = toc(t_m4_start);

fprintf('  M4 final cost=$%.4f, shed=%.4f MW, iters=%d, time=%.2fs\n', ...
    fval_m4*baseMVA, sum(x_opt4(idx.pcut))*baseMVA, ni4, t_m4);

%% ==================== 4. 各模型潮流结果 ====================
[V_m1, ~,~, S_m1] = sf.calc_pf(x_m1,   params);
[V_m2, ~,~, S_m2] = sf.calc_pf(x_m2,   params);
[V_m3, ~,~, S_m3] = sf.calc_pf(x_opt3, params);
[V_m4, ~,~, S_m4] = sf.calc_pf(x_opt4, params);

%% ==================== 5. 蒙特卡洛验证 ====================
fprintf('\n[7] 蒙特卡洛验证 (10000次)...\n');
n_mc = 10000; rng(1001);
n_model = 4;
x_sol = {x_m1, x_m2, x_opt3, x_opt4};
names = {'M1-Det', 'M2-Moment', 'M3-KDE-1side', 'M4-KDE-2side'};
results = cell(n_model, 1);

xi_mc_all = zeros(num_load + num_DG, n_mc);
for i = 1:num_load
    xi_mc_all(i,:) = sigma_load * P_load(load_buses(i)) * randn(1, n_mc);
end
for i = 1:num_DG
    xi_mc_all(num_load+i,:) = sigma_DG * P_DG_forecast(i) * randn(1, n_mc);
end

for mdl = 1:n_model
    xm = x_sol{mdl}; lam = xm(idx.lambda);
    vd = 0; bd = 0; cd = 0;
    for mc_i = 1:n_mc
        xi_mc = xi_mc_all(:, mc_i);
        pn = zeros(num_bus,1); qn = zeros(num_bus,1);
        for i = 1:num_load
            bus = load_buses(i);
            pn(bus) = P_load(bus) - xm(idx.pcut(i)) + xi_mc(i);
            qn(bus) = Q_load(bus) - xm(idx.qcut(i)) + xi_mc(i)*tan_theta;
        end
        for i = 1:num_DG
            bus = DG_buses(i);
            pn(bus) = pn(bus) - P_DG_forecast(i)*(1-lam(i)) + (1-lam(i))*xi_mc(num_load+i);
        end
        vv = false;
        for i = 2:num_bus
            Vi = V0 - sum(pn(2:end).*R_mat(i,2:end)' + qn(2:end).*X_mat(i,2:end)');
            if Vi < V_min-1e-6 || Vi > V_max+1e-6, vv = true; break; end
        end
        bb = false;
        for br = 1:num_branch
            ds = W{to_bus(br)};
            if sqrt(sum(pn(ds))^2 + sum(qn(ds))^2) > S_max_br(br)+1e-6, bb = true; break; end
        end
        dp = sum(xi_mc(1:num_load)) - sum((1-lam).*xi_mc(num_load+1:end));
        cc = eta1*max(xm(idx.pplus)+dp,0) - eta2*max(xm(idx.pminus)-dp,0) + eta3*sum(xm(idx.pcut));
        vd = vd+vv; bd = bd+bb; cd = cd+cc;
    end
    results{mdl}.vr = 100*vd/n_mc;
    results{mdl}.br = 100*bd/n_mc;
    results{mdl}.cm = cd/n_mc;
    results{mdl}.tv = max(results{mdl}.vr, results{mdl}.br);
    results{mdl}.rl = 100 - results{mdl}.tv;
end

%% ==================== 6. 结果汇总输出 ====================
fprintf('\n================================================================\n');
fprintf('             单时段四模型结果汇总\n');
fprintf('================================================================\n');
fprintf('%-22s %-14s %-14s %-14s %-14s\n', '指标', names{1}, names{2}, names{3}, names{4});
fprintf('%s\n', repmat('-', 1, 86));
fprintf('%-22s %-14.4f %-14.4f %-14.4f %-14.4f\n', '优化成本($)', ...
    fval_m1*baseMVA, fval_m2*baseMVA, fval_m3*baseMVA, fval_m4*baseMVA);
fprintf('%-22s %-14.4f %-14.4f %-14.4f %-14.4f\n', 'MC成本($)', ...
    results{1}.cm*baseMVA, results{2}.cm*baseMVA, ...
    results{3}.cm*baseMVA, results{4}.cm*baseMVA);
fprintf('%-22s %-14.2f %-14.2f %-14.2f %-14.2f\n', sprintf('可靠性(%%)>=%.0f%%', 100*(1-epsilon)), ...
    results{1}.rl, results{2}.rl, results{3}.rl, results{4}.rl);
fprintf('%-22s %-14.2f %-14.2f %-14.2f %-14.2f\n', sprintf('MC V 违反(%%)<%.0f%%', 100*epsilon), ...
    results{1}.vr, results{2}.vr, results{3}.vr, results{4}.vr);
fprintf('%-22s %-14.2f %-14.2f %-14.2f %-14.2f\n', sprintf('MC Br违反(%%)<%.0f%%', 100*epsilon), ...
    results{1}.br, results{2}.br, results{3}.br, results{4}.br);
fprintf('%-22s %-14.4f %-14.4f %-14.4f %-14.4f\n', '切负荷(MW)', ...
    sum(x_m1(idx.pcut))*baseMVA, sum(x_m2(idx.pcut))*baseMVA, ...
    sum(x_opt3(idx.pcut))*baseMVA, sum(x_opt4(idx.pcut))*baseMVA);
fprintf('%-22s %-14.4f %-14.4f %-14.4f %-14.4f\n', 'DG削减(MW)', ...
    sum(x_m1(idx.lambda).*P_DG_forecast)*baseMVA, ...
    sum(x_m2(idx.lambda).*P_DG_forecast)*baseMVA, ...
    sum(x_opt3(idx.lambda).*P_DG_forecast)*baseMVA, ...
    sum(x_opt4(idx.lambda).*P_DG_forecast)*baseMVA);
fprintf('%-22s %-14.4f %-14.4f %-14.4f %-14.4f\n', '最低电压(p.u.)', ...
    min(V_m1(2:end)), min(V_m2(2:end)), min(V_m3(2:end)), min(V_m4(2:end)));
fprintf('%-22s %-14.4f %-14.4f %-14.4f %-14.4f\n', '最高电压(p.u.)', ...
    max(V_m1(2:end)), max(V_m2(2:end)), max(V_m3(2:end)), max(V_m4(2:end)));
fprintf('%-22s %-14.2f %-14.2f %-14.2f %-14.2f\n', '最大支路负载(%)', ...
    100*max(S_m1./S_max_br), 100*max(S_m2./S_max_br), ...
    100*max(S_m3./S_max_br), 100*max(S_m4./S_max_br));
fprintf('%-22s %-14.2f %-14.2f %-14.2f %-14.2f\n', '求解时间(s)', t_m1, t_m2, t_m3, t_m4);
fprintf('================================================================\n');

%% 保存
save_fn = 'single_period_results.mat';
save(save_fn, ...
    'x_m1','x_m2','x_opt3','x_opt4', ...
    'fval_m1','fval_m2','fval_m3','fval_m4', ...
    'history3','history4', ...
    'results','names', ...
    'V_m1','V_m2','V_m3','V_m4', ...
    'S_m1','S_m2','S_m3','S_m4', ...
    't_m1','t_m2','t_m3','t_m4');
fprintf('\n  结果保存到 %s\n', save_fn);

%% ==================== 7. 绘图 ====================
clrs = [0.2 0.4 0.8;
        0.2 0.7 0.3;
        0.9 0.5 0.0;
        0.8 0.2 0.2];

figure('Name', 'Cutting-Plane Convergence');
subplot(1,2,1);
yyaxis left;
plot(1:length(history3.obj), history3.obj, '-o','LineWidth',1.8);
ylabel('Objective');
yyaxis right;
semilogy(1:length(history3.max_viol), history3.max_viol, '-s','LineWidth',1.8);
ylabel('max F_k');
xlabel('Iteration');
title('M3 (KDE-1side) Convergence');
grid on;

subplot(1,2,2);
yyaxis left;
plot(1:length(history4.obj), history4.obj, '-o','LineWidth',1.8);
ylabel('Objective');
yyaxis right;
semilogy(1:length(history4.max_viol), history4.max_viol, '-s','LineWidth',1.8);
ylabel('max F_k');
xlabel('Iteration');
title('M4 (KDE-2side) Convergence');
grid on;

figure('Name','Voltage Profile');
hold on;
h_lines = gobjects(4,1);
h_lines(1) = plot(1:num_bus, V_m1, '-', 'LineWidth',1.5, 'Color', clrs(1,:));
h_lines(2) = plot(1:num_bus, V_m2, '-', 'LineWidth',1.5, 'Color', clrs(2,:));
h_lines(3) = plot(1:num_bus, V_m3, '-', 'LineWidth',1.5, 'Color', clrs(3,:));
h_lines(4) = plot(1:num_bus, V_m4, '-', 'LineWidth',1.5, 'Color', clrs(4,:));
h_lim = plot([1 num_bus], [V_min V_min], 'k--', [1 num_bus], [V_max V_max], 'k--');
xlabel('Bus'); ylabel('V (p.u.)');
legend([h_lines; h_lim(1)], [names, {'Limit'}], 'Location','best');
title('Voltage Profile (Four Models)'); grid on; hold off;

figure('Name','DG Curtailment');
lam_mat = [x_m1(idx.lambda), x_m2(idx.lambda), x_opt3(idx.lambda), x_opt4(idx.lambda)];
hb = bar(lam_mat);
for k = 1:4, hb(k).FaceColor = clrs(k,:); end
set(gca, 'XTickLabel', arrayfun(@(b)sprintf('Bus%d',b), DG_buses, 'UniformOutput', false));
ylabel('\lambda (curtailment ratio)');
legend(names, 'Location','best');
title('DG Curtailment'); grid on;

figure('Name','Load Shedding');
shed_total = [sum(x_m1(idx.pcut)), sum(x_m2(idx.pcut)), ...
              sum(x_opt3(idx.pcut)), sum(x_opt4(idx.pcut))]*baseMVA;
hb = bar(shed_total);
hb.FaceColor = 'flat';
for k = 1:4, hb.CData(k,:) = clrs(k,:); end
set(gca, 'XTickLabel', names);
ylabel('Total Load Shedding (MW)');
title('Total Load Shedding (Four Models)'); grid on;

figure('Name','Cost Comparison');
cost_opt = [fval_m1, fval_m2, fval_m3, fval_m4]*baseMVA;
cost_mc  = [results{1}.cm, results{2}.cm, results{3}.cm, results{4}.cm]*baseMVA;
hb = bar([cost_opt; cost_mc]');
hb(1).FaceColor = [0.3 0.5 0.8];
hb(2).FaceColor = [0.9 0.6 0.2];
set(gca, 'XTickLabel', names);
ylabel('Cost ($)');
legend({'Optimization', 'Monte Carlo'}, 'Location','best');
title('Cost Comparison'); grid on;

figure('Name','MC Violation Comparison');
viol_mat = [results{1}.vr, results{2}.vr, results{3}.vr, results{4}.vr;
            results{1}.br, results{2}.br, results{3}.br, results{4}.br]';
hb = bar(viol_mat);
hb(1).FaceColor = [0.3 0.5 0.8];
hb(2).FaceColor = [0.8 0.3 0.3];
set(gca, 'XTickLabel', names);
ylabel('MC Violation Rate (%)');
legend({'Voltage Violation', 'Branch Violation'}, 'Location','best');
hold on;
plot([0.3, 4.7], [100*epsilon 100*epsilon], 'k--', 'LineWidth', 1.2);
title(sprintf('MC Violation Rate (\\epsilon = %.0f%%)', 100*epsilon));
grid on; hold off;

fprintf('\n  单时段实验完成!\n');
fprintf('================================================================\n');

%% ==================== 割平面主循环 ====================
function [x_opt, fval, ni, history] = run_cutplane( ...
        x_init, params, lb_all, ub_all, opts_main, opts_alt, cg, sf, mode)

    if strcmp(mode, 'oneside')
        obj_fn  = @(xx) sf.obj_dro_an_oneside(xx, params);
        eval_fn = @(xx, pp) sf.eval_cc_oneside(xx, pp, cg);
    else
        obj_fn  = @(xx) sf.obj_dro_an(xx, params);
        eval_fn = @(xx, pp) sf.eval_cc_analytical(xx, pp, cg);
    end

    max_iter      = 60;
    eps_feas      = 1e-4;
    stagnate_max  = 3;
    move_eps      = 1e-8;

    x_opt = x_init;
    fval  = obj_fn(x_opt);
    Acut = []; bcut = [];
    history.obj = [];
    history.max_viol = [];
    history.cuts = [];
    history.ef = [];

    stagnate_cnt = 0;
    last_x = x_opt;
    p = params;

    for iter = 1:max_iter
        ni = iter;

        % ---- 求解主问题 ----
        xn = x_opt; fvn = fval; ef = -99; ok = false;
        try
            [xn, fvn, ef] = fmincon(obj_fn, x_opt, ...
                Acut, bcut, [], [], lb_all, ub_all, ...
                @(xx) sf.con_master(xx, p), opts_main);
            ok = all(isfinite(xn));
        catch
            ok = false;
        end
        if ~ok || ef == -2
            try
                [xn2, fvn2, ef2] = fmincon(obj_fn, x_opt, ...
                    Acut, bcut, [], [], lb_all, ub_all, ...
                    @(xx) sf.con_master(xx, p), opts_alt);
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
                accept = true;
            elseif (fvn <= fval + 1e-6) || ef > 0
                accept = true;
            end
        end
        if accept
            x_opt = xn; fval = fvn;
        end

        % ---- 评估约束违反 ----
        lam_cur = x_opt(p.idx.lambda);
        [hV, hB] = sf.bw_all(lam_cur, p);
        p.hV = hV; p.hB = hB;
        [Fk, gk, vio] = eval_fn(x_opt, p);
        max_Fk = max(Fk); nvio = sum(vio);

        history.obj(end+1)      = fval;
        history.max_viol(end+1) = max_Fk;
        history.cuts(end+1)     = size(Acut, 1);
        history.ef(end+1)       = ef;

        fprintf('  iter=%2d  obj=%.6f  maxFk=%.3e  violated=%4d  cuts=%5d  ef=%d  acc=%d\n', ...
            iter, fval, max_Fk, nvio, size(Acut,1), ef, accept);

        if nvio == 0 || max_Fk <= eps_feas
            fprintf('\n*** 收敛! (max Fk = %.2e) ***\n', max_Fk);
            return;
        end

        % ---- 进展检测 ----
        if norm(x_opt - last_x, inf) < move_eps
            stagnate_cnt = stagnate_cnt + 1;
        else
            stagnate_cnt = 0;
        end
        last_x = x_opt;

        % ---- 应急: 主问题不动 -> 沿最违反约束梯度做投影下降 ----
        if stagnate_cnt >= stagnate_max
            fprintf('  [stagnation] 主问题连续%d次无进展, 触发应急投影下降...\n', ...
                stagnate_cnt);
            [~, kw] = max(Fk);
            dir = -gk(:, kw);
            if norm(dir, inf) < 1e-12
                fprintf('  [stagnation] 梯度退化, 提前结束\n');
                return;
            end
            dir = dir / max(1, norm(dir, inf));
            alpha = 1.0;
            improved = false;
            for ls = 1:25
                x_try = max(lb_all, min(ub_all, x_opt + alpha * dir));
                % 修正功率平衡等式
                [pn_t, ~] = sf.calc_ni(x_try, p);
                imb = sum(pn_t(2:end));
                x_try(p.idx.pplus)  = max(0, min(ub_all(p.idx.pplus),  imb));
                x_try(p.idx.pminus) = max(0, min(ub_all(p.idx.pminus), -imb));
                lam_t = x_try(p.idx.lambda);
                [hV_t, hB_t] = sf.bw_all(lam_t, p);
                p2 = p; p2.hV = hV_t; p2.hB = hB_t;
                [Fk_t, ~, ~] = eval_fn(x_try, p2);
                if max(Fk_t) < max_Fk * 0.999
                    x_opt = x_try; fval = obj_fn(x_opt);
                    improved = true;
                    fprintf('  [stagnation] 投影下降成功, alpha=%.4f, 新maxFk=%.3e\n', ...
                        alpha, max(Fk_t));
                    break;
                end
                alpha = alpha * 0.5;
            end
            stagnate_cnt = 0;
            if ~improved
                fprintf('  [stagnation] 投影下降失败, 提前结束\n');
                return;
            end
            continue;
        end

        % ---- 添加 cut: ak * x <= bk  ----
        vv = find(vio);
        for k = vv'
            ak = gk(:,k)';
            bk = -Fk(k) + ak * x_opt;
            if any(~isfinite(ak)) || ~isfinite(bk), continue; end
            nr = norm(ak, inf);
            if nr < 1e-12, continue; end
            ak = ak / nr;
            bk = bk / nr;
            Acut = [Acut; ak]; bcut = [bcut; bk];
        end

        % ---- 周期性清理松弛 cut ----
        if mod(iter, 5) == 0 && size(Acut, 1) > 200
            slack = bcut - Acut * x_opt;
            keep  = slack < 1e-3;
            keep(max(1, end-100):end) = true;
            Acut = Acut(keep, :); bcut = bcut(keep);
        end
        if size(Acut,1) > 3000
            Acut = Acut(end-2999:end,:); bcut = bcut(end-2999:end);
        end
    end
end

%% ==================== 本地工具 ====================
function path = local_path_to_root(node, fb, tb)
    path=[]; c=node;
    while c>1
        br=find(tb==c,1);
        if isempty(br), break; end
        path=[path,br]; c=fb(br);
    end
end

function ds = local_downstream(node, fb, tb)
    ds=node; q=node;
    while ~isempty(q)
        c=q(1); q(1)=[];
        ch=tb(fb==c);
        for i=1:length(ch)
            if ~ismember(ch(i),ds)
                ds=[ds,ch(i)]; q=[q,ch(i)];
            end
        end
    end
end