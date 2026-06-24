%% run_exp5_outofsample.m
% Out-of-Sample: 四模型 x 四分布 (Gaussian/Logistic/Laplace/Uniform)
% 输出包含: 优化成本、MC成本、可靠性、平均违反率、最大违反率

clear; clc; close all;
addpath('D:/C盘来的/casadi-windows-matlabR2016a-v3.5.5');
fprintf('================================================================\n');
fprintf('  Out-of-Sample: 四模型 x 四分布 \n ');
fprintf('================================================================\n');

try
    import casadi.*
    casadi.SX.sym('t_chk');
catch
    error('CasADi 未加载');
end

if ~exist('init_data.mat','file'), run('init_data.m'); end
load('init_data.mat');

sf = shared_functions();

n_mc = 5000;
dist_names = {'Gaussian', 'Logistic', 'Laplace', 'Uniform'};
n_dist = 4;
model_names = {'M1-Det', 'M2-Moment', 'M3-KDE-1side', 'M4-KDE-2side'};
n_model = 4;

xi_gaussian_baseline = xi_all;
Sigma_baseline = Sigma_all;

% ---------- 结果统计矩阵 ----------
oos_opt_cost     = zeros(n_dist, n_model);
oos_mc_cost      = zeros(n_dist, n_model);
oos_reliability  = zeros(n_dist, n_model);
oos_shed         = zeros(n_dist, n_model);
oos_dgcut        = zeros(n_dist, n_model);
oos_time         = zeros(n_dist, n_model);
oos_avg_viol     = zeros(n_dist, n_model);   % 平均违反率(%)
oos_max_viol     = zeros(n_dist, n_model);   % 最大违反率(%)
oos_avg_V_viol   = zeros(n_dist, n_model);   % 平均电压违反率(%)
oos_max_V_viol   = zeros(n_dist, n_model);   % 最大电压违反率(%)
oos_avg_BC_viol  = zeros(n_dist, n_model);   % 平均支路违反率(%)
oos_max_BC_viol  = zeros(n_dist, n_model);   % 最大支路违反率(%)

for di = 1:n_dist
    dname = dist_names{di};
    fprintf('\n========== 分布 %d/%d: %s ==========\n', di, n_dist, dname);
    
    % 生成样本
    if strcmp(dname, 'Gaussian')
        xi_all_di = xi_gaussian_baseline;
        Sigma_all_di = Sigma_baseline;
    else
        n_total = 10000;
        xi_all_di = cell(T_periods, 1);
        Sigma_all_di = cell(T_periods, 1);
        rng(42 + di*100);
        for t = 1:T_periods
            n_xi = num_load + num_DG;
            xi_total = zeros(n_xi, n_total);
            for i = 1:num_load
                rv = P_load_t(load_buses(i), t);
                if rv > 1e-10
                    xi_total(i,:) = gen_dist_samples(dname, sigma_std * rv, n_total);
                end
            end
            for i = 1:num_DG
                rv = P_DG_forecast(i, t);
                if rv > 1e-10
                    xi_total(num_load+i,:) = gen_dist_samples(dname, sigma_std * rv, n_total);
                end
            end
            sel_idx = randperm(n_total, num_samples);
            xi_all_di{t} = xi_total(:, sel_idx);
            Sigma_diag_t = zeros(n_xi, 1);
            for i = 1:num_load
                rv = P_load_t(load_buses(i), t);
                Sigma_diag_t(i) = (sigma_std * rv)^2;
            end
            for i = 1:num_DG
                rv = P_DG_forecast(i, t);
                Sigma_diag_t(num_load+i) = (sigma_std * rv)^2;
            end
            Sigma_all_di{t} = Sigma_diag_t;
        end
    end
    
    fprintf('  [优化] 求解四个模型...\n');
    x_all_di = cell(n_model, T_periods);
    cost_all_di = zeros(n_model, T_periods);
    time_di = zeros(n_model, 1);
    
    for t = 1:T_periods
        if mod(t,6)==1, fprintf('    t=%2d/%d\n', t, T_periods); end
        
        pt = sf.build_params(t, P_load_t, Q_load_t, P_DG_forecast, xi_all_di{t}, ...
            num_bus, num_branch, num_load, num_DG, num_samples, num_segments, ...
            load_buses, DG_buses, from_bus, to_bus, W, S_max_br, ...
            a_s, b_s, c_s, V0, V_min, V_max, tan_theta, ...
            epsilon, tau, tau_obj, eta1, eta2, eta3, baseMVA, ...
            R_mat, X_mat, A_V_load_base, A_V_DG_base, ...
            A_BC_load_base, A_BC_DG_base, A_obj_load_base, A_obj_DG_base, ...
            BC_info, n_V_constr, n_BC_constr, idx, Sigma_all_di{t}, DG_types);
        cg = build_casadi_grads(pt);
        
        lb_t = zeros(nx_per_t, 1); ub_t = inf(nx_per_t, 1);
        for i = 1:num_load
            ub_t(idx.pcut(i)) = pt.P_L(load_buses(i));
            ub_t(idx.qcut(i)) = pt.Q_L(load_buses(i));
        end
        ub_t(idx.lambda) = 1; ub_t(idx.pplus) = 10; ub_t(idx.pminus) = 10;
        
        x0t = zeros(nx_per_t, 1);
        nl_v = sum(pt.P_L(2:end)) - sum(pt.P_DG);
        x0t(idx.pplus) = max(0, nl_v); x0t(idx.pminus) = max(0, -nl_v);
        
        opts_f = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
            'MaxIterations', 500, 'OptimalityTolerance', 1e-6, 'ConstraintTolerance', 1e-6);
        
        % M1
        tic_m = tic;
        try
            [x_d, f_d] = fmincon(@(xx) sf.obj_det(xx, pt), x0t, [], [], [], [], ...
                lb_t, ub_t, @(xx) sf.con_det(xx, pt), opts_f);
        catch
            x_d = x0t; f_d = sf.obj_det(x0t, pt);
        end
        time_di(1) = time_di(1) + toc(tic_m);
        x_all_di{1,t} = x_d; cost_all_di(1,t) = f_d;
        
        % M2
        tic_m = tic;
        try
            [x_h, f_h, ~] = sf.solve_moment_DRCCO_t(pt);
        catch
            x_h = x0t; f_h = sf.obj_det(x0t, pt);
        end
        time_di(2) = time_di(2) + toc(tic_m);
        x_all_di{2,t} = x_h; cost_all_di(2,t) = f_h;
        
        % M3
        pt3 = pt; pt3.use_kde = true;
        tic_m = tic;
        try
            [x_k3, f_k3, ~] = sf.solve_cp_oneside(x_d, pt3, lb_t, ub_t, opts_f, cg);
        catch
            x_k3 = x_d; f_k3 = f_d;
        end
        time_di(3) = time_di(3) + toc(tic_m);
        x_all_di{3,t} = x_k3; cost_all_di(3,t) = f_k3;
        
        % M4
        pt4 = pt; pt4.use_kde = true;
        tic_m = tic;
        try
            [x_k4, f_k4, ~] = sf.solve_cp_analytical(x_d, pt4, lb_t, ub_t, opts_f, cg);
        catch
            x_k4 = x_d; f_k4 = f_d;
        end
        time_di(4) = time_di(4) + toc(tic_m);
        x_all_di{4,t} = x_k4; cost_all_di(4,t) = f_k4;
    end
    
    fprintf('  [MC验证]...\n');
    rng(300 + di);
    mc_V_di = zeros(n_model, T_periods);
    mc_BC_di = zeros(n_model, T_periods);
    mc_cost_di = zeros(n_model, T_periods);
    
    for t = 1:T_periods
        PL_t_v = P_load_t(:, t); QL_t_v = Q_load_t(:, t); PDG_t_v = P_DG_forecast(:, t);
        for mdl = 1:n_model
            xm = x_all_di{mdl, t};
            lam = xm(idx.lambda);
            vd=0; bd=0; cd=0;
            for mc_i = 1:n_mc
                xi_mc = zeros(num_load + num_DG, 1);
                for i = 1:num_load
                    rv = PL_t_v(load_buses(i));
                    if rv > 1e-10
                        xi_mc(i) = gen_dist_samples(dname, sigma_std * rv, 1);
                    end
                end
                for i = 1:num_DG
                    rv = PDG_t_v(i);
                    if rv > 1e-10
                        xi_mc(num_load+i) = gen_dist_samples(dname, sigma_std * rv, 1);
                    end
                end
                pn = zeros(num_bus, 1); qn = zeros(num_bus, 1);
                for i = 1:num_load
                    bus = load_buses(i);
                    pn(bus) = PL_t_v(bus) - xm(idx.pcut(i)) + xi_mc(i);
                    qn(bus) = QL_t_v(bus) - xm(idx.qcut(i)) + xi_mc(i) * tan_theta;
                end
                for i = 1:num_DG
                    bus = DG_buses(i);
                    pn(bus) = pn(bus) - PDG_t_v(i)*(1-lam(i)) + (1-lam(i))*xi_mc(num_load+i);
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
            mc_V_di(mdl,t) = 100*vd/n_mc;
            mc_BC_di(mdl,t) = 100*bd/n_mc;
            mc_cost_di(mdl,t) = cd/n_mc;
        end
    end
    
    for mdl = 1:n_model
        oos_opt_cost(di, mdl) = sum(cost_all_di(mdl,:)) * baseMVA * Delta_t;
        oos_mc_cost(di, mdl)  = sum(mc_cost_di(mdl,:)) * baseMVA * Delta_t;
        % 整体违反率(取V/Br逐时段较大值后做统计)
        max_viol_t = max(mc_V_di(mdl,:), mc_BC_di(mdl,:));
        oos_reliability(di, mdl) = 100 - mean(max_viol_t);
        oos_avg_viol(di, mdl)    = mean(max_viol_t);
        oos_max_viol(di, mdl)    = max(max_viol_t);
        % 分别记录电压/支路统计
        oos_avg_V_viol(di, mdl)  = mean(mc_V_di(mdl,:));
        oos_max_V_viol(di, mdl)  = max(mc_V_di(mdl,:));
        oos_avg_BC_viol(di, mdl) = mean(mc_BC_di(mdl,:));
        oos_max_BC_viol(di, mdl) = max(mc_BC_di(mdl,:));
        sh_i = 0; dg_i = 0;
        for t = 1:T_periods
            sh_i = sh_i + sum(x_all_di{mdl,t}(idx.pcut)) * baseMVA;
            dg_i = dg_i + sum(x_all_di{mdl,t}(idx.lambda) .* P_DG_forecast(:,t)) * baseMVA;
        end
        oos_shed(di, mdl) = sh_i;
        oos_dgcut(di, mdl) = dg_i;
        oos_time(di, mdl) = time_di(mdl);
    end
    
    fprintf('  %s: M3=$%.2f(rel=%.1f%%, maxV=%.2f%%) M4=$%.2f(rel=%.1f%%, maxV=%.2f%%)\n', ...
        dname, ...
        oos_opt_cost(di,3), oos_reliability(di,3), oos_max_viol(di,3), ...
        oos_opt_cost(di,4), oos_reliability(di,4), oos_max_viol(di,4));
end

% =================== 结果输出 ===================
fprintf('\n================================================================\n');
fprintf('  Out-of-Sample Performance (汇总, 平均/最大违反率单位 %%)\n');
fprintf('================================================================\n');
fprintf('  %-12s| %-14s | %10s | %9s | %6s | %7s | %7s\n', ...
    'Distribution','Model','OptCost($)','MCCost($)','Rel(%)','AvgVio','MaxVio');
fprintf('  %s\n', repmat('-', 1, 90));
for di = 1:n_dist
    for mdl = 1:n_model
        lbl = '';
        if mdl == 1, lbl = dist_names{di}; end
        fprintf('  %-12s| %-14s | %10.2f | %9.2f | %6.2f | %7.2f | %7.2f\n', ...
            lbl, model_names{mdl}, ...
            oos_opt_cost(di,mdl), oos_mc_cost(di,mdl), ...
            oos_reliability(di,mdl), ...
            oos_avg_viol(di,mdl), oos_max_viol(di,mdl));
    end
    if di < n_dist, fprintf('  %s\n', repmat('-', 1, 90)); end
end

fprintf('\n----- 分项违反率 (V=Voltage / BC=Branch) -----\n');
fprintf('  %-12s| %-14s | %7s | %7s | %7s | %7s\n', ...
    'Distribution','Model','AvgV(%)','MaxV(%)','AvgBC(%)','MaxBC(%)');
fprintf('  %s\n', repmat('-', 1, 80));
for di = 1:n_dist
    for mdl = 1:n_model
        lbl = '';
        if mdl == 1, lbl = dist_names{di}; end
        fprintf('  %-12s| %-14s | %7.2f | %7.2f | %7.2f | %7.2f\n', ...
            lbl, model_names{mdl}, ...
            oos_avg_V_viol(di,mdl), oos_max_V_viol(di,mdl), ...
            oos_avg_BC_viol(di,mdl), oos_max_BC_viol(di,mdl));
    end
    if di < n_dist, fprintf('  %s\n', repmat('-', 1, 80)); end
end

% =================== 保存 ===================
save_fn = 'exp5_results.mat';
save(save_fn, 'dist_names','model_names', ...
    'oos_opt_cost','oos_mc_cost','oos_reliability', ...
    'oos_shed','oos_dgcut','oos_time', ...
    'oos_avg_viol','oos_max_viol', ...
    'oos_avg_V_viol','oos_max_V_viol', ...
    'oos_avg_BC_viol','oos_max_BC_viol');
fprintf('\n  结果保存到 %s\n', save_fn);

% =================== 绘图 ===================
clrs = [0.2 0.4 0.8; 0.2 0.7 0.3; 0.9 0.5 0.0; 0.8 0.2 0.2];

% 图1: 可靠性
figure('Name','OOS Reliability');
hb = bar(oos_reliability);
for k = 1:n_model, hb(k).FaceColor = clrs(k,:); end
set(gca, 'XTickLabel', dist_names);
ylabel('Reliability (%)');
title('OOS Reliability');
legend(model_names,'Location','best'); grid on; hold on;
plot([0.3 n_dist+0.7], [100*(1-epsilon) 100*(1-epsilon)], 'k--');

% 图2: 最大违反率
figure('Name','OOS Max Violation Rate');
hb2 = bar(oos_max_viol);
for k = 1:n_model, hb2(k).FaceColor = clrs(k,:); end
set(gca, 'XTickLabel', dist_names);
ylabel('Max Violation Rate (%)');
title('Max Violation Rate (over 24h)');
legend(model_names,'Location','best'); grid on; hold on;
plot([0.3 n_dist+0.7], [100*epsilon 100*epsilon], 'k--');

% 图3: 平均违反率
figure('Name','OOS Avg Violation Rate');
hb3 = bar(oos_avg_viol);
for k = 1:n_model, hb3(k).FaceColor = clrs(k,:); end
set(gca, 'XTickLabel', dist_names);
ylabel('Average Violation Rate (%)');
title('Average Violation Rate (over 24h)');
legend(model_names,'Location','best'); grid on; hold on;
plot([0.3 n_dist+0.7], [100*epsilon 100*epsilon], 'k--');

fprintf('\n  实验5完成!\n');

%% ===== Helper =====
function samples = gen_dist_samples(dist_name, target_std, n)
    switch dist_name
        case 'Gaussian'
            samples = target_std * randn(1, n);
        case 'Logistic'
            scale = target_std * sqrt(3) / pi;
            u = rand(1, n);
            u = max(1e-15, min(1-1e-15, u));
            samples = scale * log(u ./ (1 - u));
        case 'Laplace'
            b = target_std / sqrt(2);
            u = rand(1, n) - 0.5;
            samples = -b * sign(u) .* log(1 - 2*abs(u));
        case 'Uniform'
            a = target_std * sqrt(3);
            samples = a * (2*rand(1, n) - 1);
        otherwise
            error('Unknown distribution: %s', dist_name);
    end
end