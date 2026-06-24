%% init_data.m
% 数据初始化脚本
% 运行后工作区中将包含所有系统参数、样本数据、索引结构等
% =========================================================================
clear; clc; close all;

fprintf('================================================================\n');
fprintf('  数据初始化: IEEE 33节点系统 + 24h负荷/DG曲线 + 样本生成\n');
% fprintf('  数据初始化: IEEE 123节点系统 + 24h负荷/DG曲线 + 样本生成\n');
fprintf('================================================================\n');

%% 1. 系统数据
fprintf('\n[1] 读取系统数据...\n');
mpc = loadcase('case33bw');
% mpc = loadcase('grid_IEEE123');
% grid_IEEE123 is cited from：
% L. Bobo, A. Venzke, S. Chatzivasileiadis, "Second-Order Cone Relaxations
% of the Optimal Power Flow for Active Distribution Grids", 2020. Available
% online: https://arxiv.org/abs/2001.00898
%
% W. H. Kersting, “Radial distribution test feeders,” in Conference
% Proceedings of the 2001 IEEE Power Engineering Society Winter Meeting,
% 2001. vol. 2, pp. 908–912
baseMVA = mpc.baseMVA;
num_bus = size(mpc.bus, 1);
num_branch = size(mpc.branch, 1);
bus_data = mpc.bus;
branch_data = mpc.branch;
from_bus = branch_data(:, 1);
to_bus = branch_data(:, 2);
r = branch_data(:, 3);
x = branch_data(:, 4);
rateA = branch_data(:, 6);
rateA_pu = rateA / baseMVA;

V0 = 1.0; V_max = 1.05; V_min = 0.95;
power_factor = 0.85;
tan_theta = tan(acos(power_factor));

eta1 = 90;  eta2 = 45;  eta3 = 300;
Delta_t = 1;

epsilon = 0.05;
tau = 0.01;
tau_obj = 0.01;
num_samples = 100;
num_segments = 12;
T_periods = 24;

fprintf('  节点: %d, 支路: %d, 时段: %d\n', num_bus, num_branch, T_periods);

a_s = [1,1,0.2679,-0.2679,-1,-1,-1,-1,-0.2679,0.2679,1,1];
b_s = [0.2679,1,1,1,1,0.2679,-0.2679,-1,-1,-1,-1,-0.2679];
c_s = -[-1,-1.366,-1,-1,-1.366,-1,-1,-1.366,-1,-1,-1.366,-1];

%% 2. 24h 负荷与DG曲线
fprintf('\n[2] 设置24小时负荷与DG曲线...\n');
P_load_base = max(0, bus_data(:, 3) / baseMVA);
Q_load_base = max(0, bus_data(:, 4) / baseMVA);
total_load_base = sum(P_load_base);

load_scale = 0.5*1.7*[0.40, 0.35, 0.30, 0.30, 0.35, 0.40, ...
              0.45, 0.45, 0.50, 0.55, 0.75, 0.80, ...
              0.45, 0.40, 0.40, 0.45, 0.60, 0.85, ...
              1.35, 1.40, 1.25, 1.15, 0.85, 0.75];
pv_scale = 1.3*[0.00, 0.00, 0.00, 0.00, 0.00, 0.02, ...
            0.08, 0.20, 0.45, 0.70, 1.00, 1.20, ...
            1.15, 0.95, 0.70, 0.50, 0.25, 0.05, ...
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00];
wind_scale = 0.5*1.1*[0.95, 0.70, 0.85, 0.90, 0.85, 0.70, ...
              0.45, 0.35, 0.30, 0.60, 0.65, 0.75, ...
              0.90, 0.90, 0.85, 0.90, 0.85, 0.65, ...
              0.65, 0.70, 0.75, 0.65, 0.60, 0.55];

DG_buses = [18, 22, 25, 33];
DG_types = {'wind', 'pv', 'wind', 'pv'};
num_DG = length(DG_buses);
P_DG_rated = 6*[0.2; 0.2; 0.2; 0.2] / baseMVA;%IEEE33


% load_scale = 1.5*[0.40, 0.35, 0.30, 0.30, 0.35, 0.40, ...
%               0.45, 0.45, 0.50, 0.55, 0.75, 0.80, ...
%               0.45, 0.40, 0.40, 0.45, 0.60, 0.95, ...
%               1.45, 1.40, 1.35, 1.15, 0.85, 0.75];
% pv_scale = 1.1*[0.00, 0.00, 0.00, 0.00, 0.00, 0.02, ...
%             0.08, 0.20, 0.45, 0.70, 1.00, 1.20, ...
%             1.15, 0.95, 0.70, 0.50, 0.25, 0.05, ...
%             0.00, 0.00, 0.00, 0.00, 0.00, 0.00];
% wind_scale = [0.95, 0.70, 0.85, 0.90, 0.85, 0.70, ...
%               0.45, 0.35, 0.30, 0.60, 0.65, 0.75, ...
%               0.90, 0.90, 0.85, 0.90, 0.85, 0.65, ...
%               0.65, 0.70, 0.75, 0.65, 0.60, 0.55];%IEEE123

% DG_buses = [30, 60, 80, 100, 114]; 
% DG_types = {'pv','wind','pv','wind','pv'};
% num_DG = length(DG_buses);
% P_DG_rated = 5.5*[0.3; 0.3; 0.3; 0.3; 0.3] / baseMVA; %IEEE123
P_DG_forecast = zeros(num_DG, T_periods);
for i = 1:num_DG
    for t = 1:T_periods
        if strcmp(DG_types{i}, 'wind')
            P_DG_forecast(i, t) = P_DG_rated(i) * wind_scale(t);
        else
            P_DG_forecast(i, t) = P_DG_rated(i) * pv_scale(t);
        end
    end
end

P_load_t = zeros(num_bus, T_periods);
Q_load_t = zeros(num_bus, T_periods);
for t = 1:T_periods
    P_load_t(:, t) = load_scale(t) * P_load_base;
    Q_load_t(:, t) = load_scale(t) * Q_load_base;
end

load_buses = find(P_load_base > 1e-6);
load_buses = load_buses(load_buses ~= 1);
num_load = length(load_buses);

fprintf('  基准总负荷: %.2f MW\n', total_load_base * baseMVA);
fprintf('  DG额定总容量: %.2f MW\n', sum(P_DG_rated)*baseMVA);

%% 3. 网络矩阵
fprintf('\n[3] 计算网络矩阵...\n');
R_mat = zeros(num_bus);
X_mat = zeros(num_bus);
for i = 2:num_bus
    path_i = local_get_path_to_root(i, from_bus, to_bus);
    for j = 2:num_bus
        path_j = local_get_path_to_root(j, from_bus, to_bus);
        common = intersect(path_i, path_j);
        R_mat(i,j) = sum(r(common));
        X_mat(i,j) = sum(x(common));
    end
end

W = cell(num_bus, 1);
for j = 1:num_bus
    W{j} = local_get_downstream(j, from_bus, to_bus);
end

S_max_br = rateA_pu;
S_max_br(S_max_br < 1e-6) = 2 * total_load_base;

%% 4. 灵敏度矩阵
fprintf('\n[4] 预计算灵敏度矩阵...\n');
n_V_constr = num_bus - 1;
n_BC_constr = num_branch * num_segments / 2;

A_V_load_base = zeros(n_V_constr, num_load);
A_V_DG_base = zeros(n_V_constr, num_DG);
for k = 1:n_V_constr
    i_bus = k + 1;
    for jj = 1:num_load
        A_V_load_base(k, jj) = -(R_mat(i_bus, load_buses(jj)) + X_mat(i_bus, load_buses(jj)) * tan_theta);
    end
    for jj = 1:num_DG
        A_V_DG_base(k, jj) = R_mat(i_bus, DG_buses(jj));
    end
end

A_BC_load_base = zeros(n_BC_constr, num_load);
A_BC_DG_base = zeros(n_BC_constr, num_DG);
BC_info = zeros(n_BC_constr, 2);
ci = 0;
for br = 1:num_branch
    downstream = W{to_bus(br)};
    for s = 1:num_segments/2
        ci = ci + 1;
        BC_info(ci, :) = [br, s];
        for kk = 1:num_load
            if ismember(load_buses(kk), downstream)
                A_BC_load_base(ci, kk) = a_s(s) + b_s(s) * tan_theta;
            end
        end
        for kk = 1:num_DG
            if ismember(DG_buses(kk), downstream)
                A_BC_DG_base(ci, kk) = -a_s(s);
            end
        end
    end
end

A_obj_load_base = ones(1, num_load);
A_obj_DG_base = -ones(1, num_DG);

%% 5. 不确定性样本生成 (高斯)
fprintf('\n[5] 生成不确定性样本...\n');

rng(42);
n_xi = num_load + num_DG;
n_total_samples = 10000;
sigma_std = 0.15;

xi_all = cell(T_periods, 1);
Sigma_all = cell(T_periods, 1);

for t = 1:T_periods
    xi_total = zeros(n_xi, n_total_samples);
    for i = 1:num_load
        rated_val = P_load_t(load_buses(i), t);
        xi_total(i,:) = sigma_std * rated_val * randn(1, n_total_samples);
    end
    for i = 1:num_DG
        rated_val = P_DG_forecast(i, t);
        xi_total(num_load+i,:) = sigma_std * rated_val * randn(1, n_total_samples);
    end
    sel_idx = randperm(n_total_samples, num_samples);
    xi_all{t} = xi_total(:, sel_idx);
    
    Sigma_diag = zeros(n_xi, 1);
    for i = 1:num_load
        rated_val = P_load_t(load_buses(i), t);
        Sigma_diag(i) = (sigma_std * rated_val)^2;
    end
    for i = 1:num_DG
        rated_val = P_DG_forecast(i, t);
        Sigma_diag(num_load+i) = (sigma_std * rated_val)^2;
    end
    Sigma_all{t} = Sigma_diag;
end

%% 6. 变量索引
n_pcut = num_load;
n_qcut = num_load;
n_lambda = num_DG;
nx_per_t = n_pcut + n_qcut + n_lambda + 2;

idx.pcut = 1:n_pcut;
idx.qcut = n_pcut + (1:n_qcut);
idx.lambda = n_pcut + n_qcut + (1:n_lambda);
idx.pplus = n_pcut + n_qcut + n_lambda + 1;
idx.pminus = n_pcut + n_qcut + n_lambda + 2;
idx.nx = nx_per_t;

fprintf('  每时段变量: %d\n', nx_per_t);

%% 保存所有数据到 MAT 文件
fprintf('\n[6] 保存数据到 init_data.mat ...\n');
% 仅保存数据, 不保存任何函数句柄
save('init_data.mat', '-v7', ...
    'mpc','baseMVA','num_bus','num_branch','bus_data','branch_data', ...
    'from_bus','to_bus','r','x','rateA','rateA_pu', ...
    'V0','V_max','V_min','power_factor','tan_theta', ...
    'eta1','eta2','eta3','Delta_t', ...
    'epsilon','tau','tau_obj','num_samples','num_segments','T_periods', ...
    'a_s','b_s','c_s', ...
    'P_load_base','Q_load_base','total_load_base', ...
    'load_scale','pv_scale','wind_scale', ...
    'DG_buses','DG_types','num_DG','P_DG_rated','P_DG_forecast', ...
    'P_load_t','Q_load_t','load_buses','num_load', ...
    'R_mat','X_mat','W','S_max_br', ...
    'n_V_constr','n_BC_constr', ...
    'A_V_load_base','A_V_DG_base', ...
    'A_BC_load_base','A_BC_DG_base','BC_info', ...
    'A_obj_load_base','A_obj_DG_base', ...
    'sigma_std','xi_all','Sigma_all', ...
    'n_pcut','n_qcut','n_lambda','nx_per_t','idx');

fprintf('  数据初始化完成!\n');
fprintf('================================================================\n');

%% ============== 局部工具函数 ==============
function path = local_get_path_to_root(node, fb, tb)
    path=[]; c=node;
    while c>1
        br=find(tb==c,1);
        if isempty(br), break; end
        path=[path,br]; c=fb(br);
    end
end

function ds = local_get_downstream(node, fb, tb)
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
