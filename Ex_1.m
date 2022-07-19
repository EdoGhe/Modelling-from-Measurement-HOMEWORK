clear all; close all;

t0 = 1845;
t1 = 1903;
dt = 2;
t = t0:dt:t1;

hare = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 ...
    100 92 70 10 11 137 137 18 22 52 83 18 10 9 65];
lynx = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 ...
    34 45 40 15 15 60 80 26 18 37 50 35 12 12 25];
X = [hare;lynx];
Xprime = X(:,2:end);
X = X(:,1:end-1);


figure, plot(hare,LineWidth=1.5),
hold on, plot(lynx,LineWidth=1.5),
grid on, legend('Hare','Lynx');

[U,Sigma,V] = svd(X,'econ');
% if error does not reduce not a problem because noisy data
% Try to force optdmd to give Real=0 also possible to discard non converged
% cases

Atilde = U'*Xprime*V/Sigma; % Similarity transform dx/dt=A*x -> dz/dt=Atilde*z
[W,Lambda] = eig(Atilde); 
mu = diag(Lambda);
omega = log(mu)/dt;

Phi = Xprime*(V/Sigma)*W; % DMD modes
alpha1 = Sigma*V(1,:)'; % ?
b = (W*Lambda)\alpha1;

y0 = Phi\X(:,1);
u_modes = zeros(size(V,2),length(t));
for ii = 1:length(t)
    u_modes(:,ii) = (y0.*exp(omega*(t(ii)-t0)));
end
u_dmd = Phi*u_modes;


hold on,plot(u_dmd(1,:),'--',Color=[0.00,0.45,0.74],LineWidth=1),
hold on, plot(u_dmd(2,:),'--',Color=[0.85,0.33,0.10],LineWidth=1);
grid on, legend('hare','lynx','hare_D_M_D','lynx_D_M_D');title('Standard DMD');
%% OPT-DMD
addpath('./optdmd-master/src');
addpath('./optdmd-master/examples');

X = [hare;lynx];
r = 2;
imode = 2;


% Evaluate the first guess
[w0,e0,b0] = optdmd(X,t-t0,r,imode);
e0 = 0+imag(e0)*1i;

% Constraints for the DMD eigenvalues and eigenvectors
% we impose real part of eigenvectors to be negative to avoid frecast to go
% to infinite
lbc = [-Inf*ones(2,1); -Inf*ones(2,1)];
% lbc = [zeros(2,1); -Inf*ones(2,1)];
ubc = [zeros(2,1); Inf*ones(2,1)];

copts = varpro_lsqlinopts('lbc',lbc,'ubc',ubc);

[w,e,b] = optdmd(X,t,r,imode,[],[],[],copts);
% [w,e,b] = optdmd(X,t,r,imode,[],[],[],[]);

t_test = t0:0.1:t1;
x1 = real(w*diag(b)*exp(e*(t_test)));

figure,plot(t,X',LineWidth=1.5),
hold on, plot(t_test,x1(1,:)',LineStyle="-.",LineWidth=1.5,Color=[0.00,0.45,0.74]);
hold on, plot(t_test,x1(2,:)',LineStyle="-.",LineWidth=1.5,Color=[0.85,0.33,0.10]);
grid on, legend('hare','lynx','hare_D_M_D','lynx_D_M_D');
title('optimized DMD');
%% optdmd bagging 
n_bags = 100; 
e_bags = 25;
i_bags = zeros(n_bags,e_bags);
t_b = i_bags;
rng(14788129);
E_i = [];
W_i = [];
B_i = [];

lbc = [-Inf*ones(2,1); -Inf*ones(2,1)];
% lbc = [zeros(2,1); -Inf*ones(2,1)];
ubc = [zeros(2,1); Inf*ones(2,1)];

copts = varpro_lsqlinopts('lbc',lbc,'ubc',ubc);


for ii = 1:1:n_bags
%     ia = randi(numel(t)-e_bags,1);
    i_bags(ii,:) = sort(randperm(numel(t),e_bags));
%     i_bags(ii,:) = ia:1:(e_bags+ia-1);
    t_b(ii,:) = t(i_bags(ii,:));
    X_b_i = X(:,i_bags(ii,:));
    X_b(:,:,ii) = X_b_i;
    [w_i,e_i,b_i] = optdmd(X_b_i,t_b(ii,:)-t0,r,imode,[],e0,[],copts);
    W_i(:,:,ii) = w_i;
    E_i(:,:,ii) = e_i;
    B_i(:,:,ii) = b_i;
    x2 = real(w_i*diag(b_i)*exp(e_i*(t_test-t0)));
    X2(:,:,ii) = x2;
end

B_i = squeeze(B_i);
E_i = squeeze(E_i);
i_b = find(real(B_i(1,:))>100);
E_i(:,i_b) = [];%E_i(:,i_b)-real(E_i(:,i_b));
W_i(:,:,i_b) = [];
B_i(:,i_b) = [];


W_mean = mean(W_i,3);%sum(W_i,3)/size(W_i,3);
W_var = sqrt(sum((W_i-W_mean).^2,3));%/size(W_i,3);
W_std = std(W_i,[],3);
E_mean = mean(E_i,2);%sum(E_i,2)/size(E_i,2);
E_var = sqrt(sum((E_i-E_mean).^2,2));%/size(E_i,2);
E_std = std(E_i,[],2);
B_mean = mean(B_i,2);%sum(B_i,2)/size(B_i,2);
B_var = sqrt(sum((B_i-B_mean).^2,2));%/size(B_i,2);
B_std = std(B_i,[],2);
X3_mean = real(W_mean*diag(B_mean)*exp(E_mean*(t_test-t0)));
X3_var = real(W_var*diag(B_var)*exp(E_var*((t_test-t0))));
X3_std = real(W_std*diag(B_std)*exp(E_std*((t_test-t0))));
% X3_mean = mean(X2,3);
% X3_std = std(X2,[],3);
% X3_var = sqrt(sum((X2-X3_mean).^2,3));
figure,plot(i_bags',LineWidth=1);
grid on,title('Bagged times');

figure, hold all,
% plot(X',LineWidth=1.5);
X2(:,:,i_b) = [];
for jj = 1:1:(n_bags-numel(i_b))
    plot(X2(1,:,jj)',LineWidth=1,LineStyle="-.",Color='b');
    hold on, plot(X2(2,:,jj)',LineWidth=1,LineStyle="-.",Color='r');
end
grid on, title('Bagged DMD');
% legend('hare','lynx','hare_D_M_D','lynx_D_M_D');

figure,plot(t_test, X3_mean',LineWidth=1.5,LineStyle="-.");
hold on, plot(t,hare,LineWidth=1.5,Color=[0.00,0.45,0.74]),
hold on, plot(t,lynx,LineWidth=1.5,Color=[0.85,0.33,0.10]),
title('Bagged DMD'),grid on,legend('Hare_D_M_D','Lynx_D_M_D','Hare','Lynx');

figure,plot(t_test,X3_std',LineWidth=1.5);
title('Var plot'),grid on,legend('Hare','Lynx');

%% Time delay
e_t = 20;
X_tde = [];
Y_tde = [];
t_tde = [];
for ii=1:1:(numel(t)-e_t+1)
    X_tde = [X_tde; X(:,ii:1:(ii+e_t-1))];
    t_tde = [t_tde;t(ii:1:(ii+e_t-1))];
    t_tde = [t_tde;t(ii:1:(ii+e_t-1))];
end

% [U,S,V] = svd(X_tde,'econ');
% 
% figure,plot(diag(S)/(sum(diag(S))));
% figure,subplot(2,1,1),plot(U(:,1:3),'LineWidth',1);
% subplot(2,1,2),plot(V(:,1:3),'LineWidth',1);

r = 10;%18;
imode = 2;
t_tde1 = t(1:1:e_t)-t0;

lbc = [-Inf*ones(r,1); -Inf*ones(r,1)];
% lbc = [zeros(r,1); -Inf*ones(r,1)];
ubc = [zeros(r,1); Inf*ones(r,1)];

copts = varpro_lsqlinopts('lbc',lbc,'ubc',ubc);
opts = varpro_opts('maxiter',1000);
[w_i,e_i,b_i] = optdmd(X_tde,t_tde1,r,imode,opts,[],[],copts);

e0 = e_i-real(e_i);

[w_i,e_i,b_i] = optdmd(X_tde,t_tde1,r,imode,opts,e0,[],copts);

Y_tde = real(w_i*diag(b_i)*exp(e_i*(t_test-t0)));

figure,
plot([t;t]',X',LineWidth=1);
hold on,plot(t_test,Y_tde(1:2,:)',LineWidth=1.5,LineStyle="--"),
colororder([0.00,0.45,0.74
            0.85,0.33,0.10 
            0.00,0.45,0.74 
            0.85,0.33,0.10]);
legend('hare','lynx','hare TD-DMD','lynx TD-DMD');
grid on,ylabel('Thousands of elements'),xlabel('Year');
%% bagged time delay
e_t = 20;%15
n_bags = 100;
i_bags = zeros(n_bags,e_t);
t_b = i_bags;
rng(14788129);
n_t = 25;%20
W_i = [];
E_i = [];
B_i = [];
X2b = [];
% figure;
opts = varpro_opts('maxiter',50);
%r = 4;%min(size(X_tb))-3;8

for ii = 1:1:n_bags
%     ia = randi(numel(t)-e_bags,1);
    i_bags(ii,:) = sort(randperm(n_t,e_t));
%     i_bags(ii,:) = ia:1:(e_bags+ia-1);
    t_b(ii,:) = t(i_bags(ii,:));
    X_b_i = X(:,i_bags(ii,:));
    X_tb = [];
    t_tb = [];
    for jj=1:1:(numel(t)-n_t+1) %+1
        X_tb = [X_tb; X(:,i_bags(jj,:)+jj-1)];
        t_tb = [t_tb;t(:,i_bags(jj,:)+jj-1)];
    end
%     t_tb_ii(ii,:) = t_tb(1,:);
%     X_b(:,:,ii) = X_b_i;
    lbc = [-Inf*ones(r,1); -Inf*ones(r,1)];
%     lbc = [zeros(r,1); -Inf*ones(r,1)];
    ubc = [zeros(r,1); Inf*ones(r,1)];
    copts = varpro_lsqlinopts('lbc',lbc,'ubc',ubc);

    imode = 2;
%     if ii==1
%         [w_0,e_0,b_0] = optdmd(X_tb,t_b(ii,:)-t0,r,imode,opts,[],[],copts);
%         e_0 = e_0-real(e_0);
%     end
    [w_i,e_i,b_i] = optdmd(X_tb,t_tb(1,:)-t0,r,imode,opts,e0,[],copts);
%     e_0 = e_0-real(e_i);

    W_i(:,:,ii) = w_i;
    E_i(:,:,ii) = e_i;
    B_i(:,:,ii) = b_i;
    x2b = real(w_i*diag(b_i)*exp(e_i*(t_test-t0+0)));
    X2b(:,:,ii) = x2b;
%     plot(x2b(1:2,:)');hold on;
%     plot(t_b(ii,:)-t0);hold on;
end

B_i_mean_c = mean(B_i,1);
i_b = find(real(B_i_mean_c(:,:))>100);
E_i(:,i_b) = [];%E_i(:,i_b)-real(E_i(:,i_b));
W_i(:,:,i_b) = [];
B_i(:,i_b) = [];

W_mean = mean(W_i,3);%sum(W_i,3)/size(W_i,3);
W_var = sqrt(sum((W_i-W_mean).^2,3));%/size(W_i,3);
W_std = std(W_i,[],3);
E_i = squeeze(E_i);
E_mean = mean(E_i,2);%sum(E_i,2)/size(E_i,2);
E_var = sqrt(sum((E_i-E_mean).^2,2));%/size(E_i,2);
E_std = std(E_i,[],2);
B_i = squeeze(B_i);
B_mean = mean(B_i,2);%sum(B_i,2)/size(B_i,2);
B_var = sqrt(sum((B_i-B_mean).^2,2));%/size(B_i,2);
B_std = std(B_i,[],2);
X3_mean = real(W_mean*diag(B_mean)*exp(E_mean*(t_test-t0)));
X3_var = real(W_var*diag(B_var)*exp(E_var*(t_test-t0)));
X3_std = real(W_std*diag(B_std)*exp(E_std*(t_test-t0)));
% X3_mean = mean(X2b,3);
% X3_std = std(X2b,[],3);
% X3_var = sqrt(sum((X2b-X3_mean).^2,3));

figure,plot(t_test,X3_mean(1:2,:)',LineWidth=1)
hold on,plot(t,X',LineStyle="--",LineWidth=1);
colororder([0.00,0.45,0.74
            0.85,0.33,0.10 
            0.00,0.45,0.74 
            0.85,0.33,0.10]);
legend('hare TD-DMD+bagging','lynx TD-DMD+baggig','hare','lynx');
grid on,ylabel('Thousands of elements'),xlabel('Year');
figure,plot(t_test,X3_std(1:2,:)',LineWidth=1)
legend('std hare','std lynx')
grid on,ylabel('Thousands of elements'),xlabel('Year');
ylim([0,round(max(max(X3_std)))])

% error_p = (X3_mean(1:2,:)-X)./X;
% error_p = sum(error_p,2)/numel(error_p(1,:));


%% Standard fitting

options = optimset('MaxFunEvals',5000);

p0(1) = hare(1);
p0(2) = lynx(1);
T = 6;
w = 2*pi/T;
p0(3) = 0.5;
p0(4) = 0.04;
p0(5) = 0.4;
p0(6) = 0.03;

[p,fval,exitflag] = ...
fminsearch(@leastcomplv,p0,options,t,hare,lynx)

p0 = p;

[p,fval,exitflag] = ...
fminsearch(@leastcomplv,p0,options,t,hare,lynx)

[tt,x_solved] = ode23(@lv_eq,t_test,[p(1) p(2)],[],p(3),p(4),p(5),p(6));

figure,plot(t,[hare;lynx],linewidth=1),hold on,plot(t_test,x_solved,linewidth=1,LineStyle="--")
colororder([0.00,0.45,0.74
            0.85,0.33,0.10 
            0.00,0.45,0.74 
            0.85,0.33,0.10]);
grid on;
legend('hare','lynx','hare_f_i_t','lynx_f_i_t');
grid on,ylabel('Thousands of elements'),xlabel('Year');

function J = leastcomplv(p,tdata,xdata,ydata)

[t1,y] = ode23(@lotvol,tdata,[p(1),p(2)],[],p(3),p(4),p(5),p(6)); 
errx = y(:,1)-xdata';
erry = y(:,2)-ydata';
J = errx'*errx + erry'*erry;
end

function dydt = lotvol(t1,y,b,p,d,r)
tmp1 = b*y(1) - p*y(1)*y(2);
tmp2 = -d*y(2) + r*y(1)*y(2);
dydt = [tmp1;tmp2];
end

function dx=lv_eq(t1,x_data,b,p,d,r)
tmp1 = b*x_data(1) - p*x_data(1)*x_data(2);
tmp2 = -d*x_data(2) +r*x_data(1)*x_data(2);
dx = [tmp1;tmp2];
end



