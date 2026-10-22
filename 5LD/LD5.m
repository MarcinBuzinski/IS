close all
clear all
clc

%% ========================================================================
% 5 laboratorinis darbas
% Diferencialinių lygčių sprendimas skaitiniais metodais
%
% RC grandinė: įėjimo įtampa Uin(t), išėjimo įtampa ant kondensatoriaus Uout(t).
% Laiko pastovioji tau = R*C.
%
% Srovė grandinėje:
%   I = (Uin - Uout) / R = C * dUout/dt
%
% Diferencialinė lygtis:
%   dUout/dt = (Uin(t) - Uout) / tau
%
% Pradinės sąlygos:
%   Uout(0) = 0
%   Uin(t) = 1,  kai 0 <= t <= 0.5 s
%   Uin(t) = 0,  kai 0.5 <= t <= 1 s
%   tau = 0.1 s
% Taškas t = 0.5 s yra perjungimo momentas: nuo jo Uin = 0.
%% ========================================================================

tau = 0.1;          % s
h   = 0.01;         % Eulerio žingsnis, s (kaip metodinėje medžiagoje)
t0  = 0;
t_switch = 0.5;     % s, čia Uin šoka iš 1 V į 0 V
t1  = 1;
U0  = 0;            % iškrautas kondensatorius

%% 1. Skaitinis sprendimas: Eulerio metodas
% Uout(t+h) = Uout(t) + h * (Uin(t) - Uout(t)) / tau
% Žingsnis, kuris prasideda ties t = 0.5, jau naudoja Uin = 0.

N = round((t1 - t0) / h) + 1;
t_euler = linspace(t0, t1, N).';
U_euler = zeros(N, 1);
U_euler(1) = U0;

for k = 1:N-1
    if t_euler(k) < t_switch
        Uin = 1;
    else
        Uin = 0;
    end
    U_euler(k+1) = U_euler(k) + h * (Uin - U_euler(k)) / tau;
end

%% 1. Skaitinis sprendimas: ode45
% Uin(t) trūki, todėl lygtis integruojama dviem lygiais intervalais.
% 0 <= t <= 0.5:  du/dt = (1 - u) / tau
% 0.5 <= t <= 1:  du/dt = (0 - u) / tau

f1 = @(t, u) (1 - u) / tau;
f2 = @(t, u) (0 - u) / tau;

[t_ode1, U_ode1] = ode45(f1, [t0 t_switch], U0);
[t_ode2, U_ode2] = ode45(f2, [t_switch t1], U_ode1(end));

t_ode = [t_ode1; t_ode2(2:end)];
U_ode = [U_ode1; U_ode2(2:end)];

%% 2. Analitinis sprendinys (dsolve), dviem intervalais
% 0 <= t <= 0.5, Uin = 1:
%   du/dt = (1 - u) / tau,   u(0) = 0
%   u(t) = 1 - exp(-t/tau)
%
% 0.5 < t <= 1, Uin = 0, pradinė sąlyga – reikšmė ties t = 0.5:
%   du/dt = -u / tau
%   u(t) = u(0.5) * exp(-(t - 0.5)/tau)

syms u(t)
tau_s = sym(1)/10;

ode1 = diff(u, t) == (1 - u) / tau_s;
u1(t) = dsolve(ode1, u(0) == 0);
u1(t) = simplify(u1);

u_pus = u1(sym(1)/2);
ode2 = diff(u, t) == (0 - u) / tau_s;
u2(t) = dsolve(ode2, u(sym(1)/2) == u_pus);
u2(t) = simplify(u2);

fprintf('Diferencialinė lygtis: dUout/dt = (Uin(t) - Uout) / tau, tau = 0.1 s\n');
fprintf('Analitinis sprendinys, 0 <= t <= 0.5 s:\n');
disp(u1);
fprintf('Analitinis sprendinys, 0.5 < t <= 1 s:\n');
disp(u2);

t_anal = t_euler;
U_anal = zeros(size(t_anal));
iki_pus = t_anal <= t_switch;
U_anal(iki_pus)  = double(u1(t_anal(iki_pus)));
U_anal(~iki_pus) = double(u2(t_anal(~iki_pus)));

% ode45 taškai perkeliami ant tos pačios laiko ašies, kad būtų galima
% nubraižyti skirtumą
U_ode_t = interp1(t_ode, U_ode, t_anal, 'pchip');

skirt_euler = U_euler - U_anal;
skirt_ode   = U_ode_t - U_anal;

fprintf('\nDidžiausias |Euler - dsolve| = %.6e V\n', max(abs(skirt_euler)));
fprintf('Didžiausias |ode45 - dsolve| = %.6e V\n', max(abs(skirt_ode)));

%% 3. Palyginimo grafikai
figure(1)
plot(t_euler, U_euler, 'b', 'LineWidth', 1.5)
hold on
plot(t_ode, U_ode, 'r--', 'LineWidth', 1.5)
plot(t_anal, U_anal, 'k:', 'LineWidth', 2)
grid on
xlabel('t, s')
ylabel('U_{out}(t), V')
title('RC grandinės išėjimo įtampa')
legend('Eulerio metodas', 'ode45', 'dsolve (analitinis)', 'Location', 'best')

figure(2)
plot(t_anal, skirt_euler, 'b', 'LineWidth', 1.5)
hold on
plot(t_anal, skirt_ode, 'r', 'LineWidth', 1.5)
grid on
xlabel('t, s')
ylabel('\Delta U_{out}(t), V')
title('Skaitinio ir analitinio sprendinių skirtumas')
legend('Euler - dsolve', 'ode45 - dsolve', 'Location', 'best')
