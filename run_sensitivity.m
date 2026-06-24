%% run_sensitivity.m
% KDE-DRCCO 参数敏感性

clear; clc; close all;
addpath('casadi path');
fprintf('================================================================\n');
fprintf('  KDE-DRCCO tau 敏感性 \n ');
fprintf('================================================================\n');

%% CasADi
try
    import casadi.*
    casadi.SX.sym('t_chk');
catch
    error('CasADi 未加载');
end

%% 加载
if ~exist('init_data.mat','file'), run('init_data.m'); end
load('init_data.mat');

sf = shared_functions();

baseline_fn = 'baseline_results.mat';
if ~exist(baseline_fn,'file')
    error('请先运行 run_baseline.m');
end
bl = load(baseline_fn);
bl = load(baseline_fn);

%% tau 扫描
tau_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0];
n_tau = length(tau_values);

exp1_cost = zeros(n_tau, 1);
exp1_reliability = zeros(n_tau, 1);
exp1_shed = zeros(n_tau, 1);
exp1_dgcut = zeros(n_tau, 1);
exp1_mc_viol_max = zeros(n_tau, 1);
exp1_mc_viol_avg = zeros(n_tau, 1);
exp1_time = zeros(n_tau, 1);
exp1_avg_iter = zeros(n_tau, 1);
exp1_mc_cost = zeros(n_tau, 1);

n_mc = 5000;

for ti = 1:n_tau
    tau_i = tau_values(ti);
    fprintf('\n  --- tau = %.5f (第%d/%d) ---\n', tau_i, ti, n_tau);
    
    tc_i = 0; ts_i = 0; tcu_i = 0;
    mc_V_i = zeros(1, T_periods); mc_BC_i = zeros(1, T_periods);
    mc_cost_i = zeros(1, T_periods);
    iter_sum = 0;
    t_start = tic;
    
    x_all_i = cell(1, T_periods);
    
    for t = 1:T_periods
        pt = sf.build_params(t, P_load_t, Q_load_t, P_DG_forecast, xi_all{t}, ...
            num_bus, num_branch, num_load, num_DG, num_samples, num_segments, ...
            load_buses, DG_buses, from_bus, to_bus, W, S_max_br, ...
            a_s, b_s, c_s, V0, V_min, V_max, tan_theta, ...
            epsilon, tau_i, tau_i, eta1, eta2, eta3, baseMVA, ...
            R_mat, X_mat, A_V_load_base, A_V_DG_base, ...
            A_BC_load_base, A_BC_DG_base, A_obj_load_base, A_obj_DG_base, ...
            BC_info, n_V_constr, n_BC_constr, idx, Sigma_all{t}, DG_types);
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

        pt4 = pt; pt4.tau = tau_i; pt4.tau_obj = tau_i; pt4.use_kde = true;
        [x_k, f_k, ni_k] = sf.solve_cp_analytical(x0t, pt4, lb_t, ub_t, opts_f, cg);
        
        x_all_i{t} = x_k;
        tc_i = tc_i + f_k * baseMVA * Delta_t;
        ts_i = ts_i + sum(x_k(idx.pcut))*baseMVA;
        tcu_i = tcu_i + sum(x_k(idx.lambda).*P_DG_forecast(:,t))*baseMVA;
        iter_sum = iter_sum + ni_k;
    end
    exp1_time(ti) = toc(t_start);
    
    % MC
    rng(123);
    for t = 1:T_periods
        PL_t_v = P_load_t(:,t); QL_t_v = Q_load_t(:,t); PDG_t_v = P_DG_forecast(:,t);
        xm = x_all_i{t}; lam = xm(idx.lambda);
        vd=0; bd=0; cd=0;
        for mc_i = 1:n_mc
            xi_mc = zeros(num_load+num_DG, 1);
            for i=1:num_load, xi_mc(i)=sigma_std*PL_t_v(load_buses(i))*randn(); end
            for i=1:num_DG, xi_mc(num_load+i)=sigma_std*PDG_t_v(i)*randn(); end
            pn=zeros(num_bus,1); qn=zeros(num_bus,1);
            for i=1:num_load
                bus=load_buses(i);
                pn(bus)=PL_t_v(bus)-xm(idx.pcut(i))+xi_mc(i);
                qn(bus)=QL_t_v(bus)-xm(idx.qcut(i))+xi_mc(i)*tan_theta;
            end
            for i=1:num_DG
                bus=DG_buses(i);
                pn(bus)=pn(bus)-PDG_t_v(i)*(1-lam(i))+(1-lam(i))*xi_mc(num_load+i);
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
        mc_V_i(t)=100*vd/n_mc; mc_BC_i(t)=100*bd/n_mc;
        mc_cost_i(t) = cd/n_mc;
    end
    
    exp1_cost(ti) = tc_i;
    exp1_reliability(ti) = 100 - mean(max(mc_V_i, mc_BC_i));
    exp1_shed(ti) = ts_i;
    exp1_dgcut(ti) = tcu_i;
    exp1_mc_viol_max(ti) = max(max(mc_V_i, mc_BC_i));
    exp1_mc_viol_avg(ti) = mean(max(mc_V_i, mc_BC_i));
    exp1_avg_iter(ti) = iter_sum / T_periods;
    exp1_mc_cost(ti) = sum(mc_cost_i) * baseMVA * Delta_t;
    
    fprintf('    Cost=$%.2f, Reliability=%.2f%%, MaxViol=%.2f%%, Time=%.1fs\n', ...
        tc_i, exp1_reliability(ti), exp1_mc_viol_max(ti), exp1_time(ti));
end

%% 参考
m1_cost_ref = bl.tc(1);
m1_rel_ref = 100 - mean(max(bl.mc_V(1,:), bl.mc_BC(1,:)));
m2_cost_ref = bl.tc(2);
m2_rel_ref = 100 - mean(max(bl.mc_V(2,:), bl.mc_BC(2,:)));

fprintf('\n  ===== 实验1 汇总表 =====\n');
fprintf('  tau        | OptCost($)  | MCCost($)   | Reliability(%%) | MaxViol(%%) | AvgIter | Time(s)\n');
for ti=1:n_tau
    fprintf('  %.5f  | %11.2f | %11.2f | %14.2f | %10.2f | %7.1f | %6.1f\n', ...
        tau_values(ti), exp1_cost(ti), exp1_mc_cost(ti), exp1_reliability(ti), ...
        exp1_mc_viol_max(ti), exp1_avg_iter(ti), exp1_time(ti));
end
fprintf('  M2 Moment  | %11.2f |             | %14.2f |\n', m2_cost_ref, m2_rel_ref);
fprintf('  M1 Det     | %11.2f |             | %14.2f |\n', m1_cost_ref, m1_rel_ref);

save_fn = 'exp1_results.mat';
save(save_fn, 'tau_values','exp1_cost','exp1_reliability','exp1_shed','exp1_dgcut',...
    'exp1_mc_viol_max','exp1_mc_viol_avg','exp1_avg_iter','exp1_time','exp1_mc_cost',...
    'm1_cost_ref','m1_rel_ref','m2_cost_ref','m2_rel_ref');
%% 绘图
figure('Name','Pareto Front');
plot(exp1_cost, exp1_reliability, '-o','LineWidth',1.8,'MarkerSize',6); hold on;
plot(m2_cost_ref, m2_rel_ref, 's','MarkerSize',10,'MarkerFaceColor','g');
plot(m1_cost_ref, m1_rel_ref, '^','MarkerSize',10,'MarkerFaceColor','b');
xlabel('24h Total Cost ($)'); ylabel('Reliability (%)');
title(sprintf('Pareto Front'));
legend('KDE-DRCCO','Moment','Det','Location','best');
grid on; box on;

figure('Name','Tau Sensitivity');
yyaxis left; semilogx(tau_values, exp1_cost, '-o','LineWidth',1.8);
ylabel('Cost ($)');
yyaxis right; semilogx(tau_values, exp1_reliability, '-o','LineWidth',1.8);
ylabel('Reliability (%)');
xlabel('\tau'); title('\tau Sensitivity'); grid on; box on;

fprintf('\n  实验1完成!\n');
fprintf('================================================================\n');
