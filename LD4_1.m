close all
clear all
clc
%% ========================================================================
% Ranka rašytų SKAITMENŲ (0-9) atpažinimas naudojant RBF neuroninį tinklą
% Palyginami DU tinklai: su 13 neuronų (originalus dydis) ir su 5 neuronais
% (sumažintas dydis), abu apmokomi tais pačiais duomenimis ir abiem
% patikrinamas atpažinimas su tuo pačiu  testiniu vaizdu.
%% ========================================================================

%% 1. Mokymo duomenų (skaitmenų pavyzdžių) nuskaitymas ir požymių skaičiavimas
pavadinimas = 'train_data_skaiciai.png';

% Jūsų vaizde yra 9 eilutės, kiekvienoje po 10 skaitmenų (0..9)
eiluciu_sk = 9;

pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, eiluciu_sk);

% požymiai iš celių masyvo perkeliami į matricą
P = cell2mat(pozymiai_tinklo_mokymui);

%% 2. Teisingų atsakymų (target) matricos sudarymas
% Skaitmenų klasių yra 10 (0,1,...,9)
klasiu_sk = 10;
T = repmat(eye(klasiu_sk), 1, eiluciu_sk);

%% 3. Dviejų RBF tinklų mokymas: su 13 ir su 5 neuronais
tinklas13 = newrb(P, T, 0, 1, 13);
tinklas5  = newrb(P, T, 0, 1, 5);

%% 4. Abiejų tinklų patikra su mokymo duomenimis (tikslumo palyginimui)
[~, teisinga_klase] = max(T);

Y13 = sim(tinklas13, P);
[~, atpazinta13] = max(Y13);
tikslumas13 = sum(atpazinta13 == teisinga_klase) / length(teisinga_klase) * 100;

Y5 = sim(tinklas5, P);
[~, atpazinta5] = max(Y5);
tikslumas5 = sum(atpazinta5 == teisinga_klase) / length(teisinga_klase) * 100;

fprintf('\nTikslumas mokymo aibėje:\n');
fprintf('  13 neuronų -> %.1f %%\n', tikslumas13);
fprintf('   5 neuronai -> %.1f %%\n', tikslumas5);

% Kiekvienai klasei (skaitmeniui 0..9) paimame DAŽNIAUSIĄ tinklo atsakymą
% iš visų tos klasės mokymo pavyzdžių - taip gauname VIENĄ apibendrintą
% spėjimą kiekvienam skaitmeniui, o ne kartojame per kiekvieną eilutę.
turi_buti = '0123456789';
speta13_klases = zeros(1, klasiu_sk);
speta5_klases  = zeros(1, klasiu_sk);

for c = 1:klasiu_sk
    idx = (teisinga_klase == c);
    speta13_klases(c) = mode(atpazinta13(idx)) - 1;
    speta5_klases(c)  = mode(atpazinta5(idx)) - 1;
end

speta13_str = sprintf('%d', speta13_klases);
speta5_str  = sprintf('%d', speta5_klases);

fprintf('\nMokymo duomenų atpažinimo palyginimas (pagal klases):\n');
fprintf('Turi buti = %s\n', turi_buti);
fprintf('13 neuronų: speta = %s\n', speta13_str);
fprintf(' 5 neuronai: speta = %s\n', speta5_str);

%% 5. Savo ranka rašyto skaičiaus (naujo, nematyto) atpažinimas
pavadinimas = 'test_skaicius1.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);
P2 = cell2mat(pozymiai_patikrai);

% --- Atpažinimas su 13 neuronų tinklu ---
Y2_13 = sim(tinklas13, P2);
[~, b13] = max(Y2_13);
atsakymas13 = [];
for k = 1:size(P2,2)
    atsakymas13 = [atsakymas13, num2str(b13(k) - 1)];
end

% --- Atpažinimas su 5 neuronų tinklu ---
Y2_5 = sim(tinklas5, P2);
[~, b5] = max(Y2_5);
atsakymas5 = [];
for k = 1:size(P2,2)
    atsakymas5 = [atsakymas5, num2str(b5(k) - 1)];
end

%% 6. Antro ranka rašyto skaičiaus (dar vieno, nematyto) atpažinimas
pavadinimas = 'test_skaicius2.png';
pozymiai_patikrai2 = pozymiai_raidems_atpazinti(pavadinimas, 1);
P3 = cell2mat(pozymiai_patikrai2);

% --- Atpažinimas su 13 neuronų tinklu ---
Y3_13 = sim(tinklas13, P3);
[~, c13] = max(Y3_13);
atsakymas13_2 = [];
for k = 1:size(P3,2)
    atsakymas13_2 = [atsakymas13_2, num2str(c13(k) - 1)];
end

% --- Atpažinimas su 5 neuronų tinklu ---
Y3_5 = sim(tinklas5, P3);
[~, c5] = max(Y3_5);
atsakymas5_2 = [];
for k = 1:size(P3,2)
    atsakymas5_2 = [atsakymas5_2, num2str(c5(k) - 1)];
end

%% 7. Galutinis rezultatas - abiejų tinklų spėjimai abiem testiniams skaičiams
fprintf('\n=== GALUTINIS REZULTATAS ===\n');
fprintf('1-as testinis skaičius:\n');
fprintf('  Spėjimas su 13 neuronų tinklu: %s\n', atsakymas13);
fprintf('  Spėjimas su  5 neuronų tinklu: %s\n', atsakymas5);
fprintf('2-as testinis skaičius:\n');
fprintf('  Spėjimas su 13 neuronų tinklu: %s\n', atsakymas13_2);
fprintf('  Spėjimas su  5 neuronų tinklu: %s\n', atsakymas5_2);

figure(8)
text(0.1, 0.8, ['1-as skaičius, 13 neuronų: ', atsakymas13], 'FontSize', 24)
text(0.1, 0.6, ['1-as skaičius,  5 neuronai: ', atsakymas5], 'FontSize', 24)
text(0.1, 0.4, ['2-as skaičius, 13 neuronų: ', atsakymas13_2], 'FontSize', 24)
text(0.1, 0.2, ['2-as skaičius,  5 neuronai: ', atsakymas5_2], 'FontSize', 24)
axis off
title('Skaičių atpažinimas: 13 vs 5 neuronų tinklas')