function pozymiai = pozymiai_raidems_atpazinti(pavadinimas, pvz_eiluciu_sk)
%%  pozymiai = pozymiai_raidems_atpazinti(pavadinimas, pvz_eiluciu_sk)
% Features = pozymiai_raidems_atpazinti(image_file_name, Number_of_symbols_lines)
% taikymo pavyzdys:
% pozymiai = pozymiai_raidems_atpazinti('test_data.png', 8); 
%% Vaizdo su pavyzdžiais nuskaitymas ir išvedimas į ekraną

V = imread(pavadinimas);
figure(12), imshow(V)
%% Raidžių iškirpimas ir sudeliojimas į kintamojo 'objektai' celes

% Spalvoto paveiklsėlio keitimas į juodai-baltą
V_pustonis = rgb2gray(V);

% vaizdo keitimo dvejetainiu slenkstinės reikšmės paieška
slenkstis = graythresh(V_pustonis);

% pustonio vaizdo keitimas dvejetainiu
V_dvejetainis = im2bw(V_pustonis,slenkstis);

% rezultato atvaizdavimas
figure(1), imshow(V_dvejetainis)

% vaizde esančių objektų kontūrų paieška
V_konturais = edge(uint8(V_dvejetainis));

% rezultato atvaizdavimas
figure(2),imshow(V_konturais)

% objektų kontūrų užpildymas 
se = strel('square',7); % struktūrinis elementas užpildymui
V_uzpildyti = imdilate(V_konturais, se); 

% rezultato atvaizdavimas
figure(3),imshow(V_uzpildyti)

% tuštumų objektų viduje užpildymas
V_vientisi= imfill(V_uzpildyti,'holes');

% rezultato atvaizdavimas
figure(4),imshow(V_vientisi)

% vientisų objektų dvejetainiame vaizde numeravimas
[O_suzymeti Skaicius] = bwlabel(V_vientisi);

% apskaičiuojami objektų dvejetainiame vaizde požymiai
O_pozymiai = regionprops(O_suzymeti);

% nuskaitomos požymių - objektų ribų koordinačių - reikšmės
O_ribos = [O_pozymiai.BoundingBox];

% kadangi ribą nusako 4 koordinatės, pergrupuojame reikšmes
O_ribos = reshape(O_ribos,[4 Skaicius]); % Skaicius - objektų skaičius

% nuskaitomos poþymių - objektų masės centro koordinačių - reikšmės
O_centras = [O_pozymiai.Centroid];

% kadangi centrą nusako 2 koordinatės, pergrupuojame reikšmes
O_centras = reshape(O_centras,[2 Skaicius]);
O_centras = O_centras';

% pridedamas kiekvienam objektui vaize numeris (trečias stulpelis šalia koordinačių)
O_centras(:,3) = 1:Skaicius;

% surûšiojami objektai pagal x koordinatę - stulpelį
O_centras = sortrows(O_centras,2);

% rūšiojama atsižvelgiant į pavyzdžių eilučių ir raidžių skaičių
raidziu_sk = Skaicius/pvz_eiluciu_sk;
for k = 1:pvz_eiluciu_sk
    O_centras((k-1)*raidziu_sk+1:k*raidziu_sk,:) = ...
        sortrows(O_centras((k-1)*raidziu_sk+1:k*raidziu_sk,:),3);
end
% iš dvejetainio vaizdo pagal objektų ribas iškerpami vaizdo fragmentai
for k = 1:Skaicius
    objektai{k} = imcrop(V_dvejetainis,O_ribos(:,O_centras(k,3)));
end

% vieno iš vaizdo fragmentų atvaizdavimas
figure(5),
for k = 1:Skaicius
   subplot(pvz_eiluciu_sk,raidziu_sk,k), imshow(objektai{k})
end

% vaizdo fragmentai apkerpami, panaikinant foną iš kraštų (pagal stačiakampį)
for k = 1:Skaicius % Skaicius = 88, jei yra 88 raidės
    V_fragmentas = objektai{k};
    % nustatomas kiekvieno vaizdo fragmento dydis
    [aukstis, plotis] = size(V_fragmentas);
    
    % 1. Baltų stulpelių naikinimas
    % stulpelių sumų skaičiavimas
    stulpeliu_sumos = sum(V_fragmentas,1);
    % naikiname tuos stulpelius, kur suma lygi aukščiui
    V_fragmentas(:,stulpeliu_sumos == aukstis) = [];
    % perskaičiuojamas objekto dydis
    [aukstis, plotis] = size(V_fragmentas);
    % 2. Baltų eilučių naikinimas
    % eilučių sumų skaičiavimas
    eiluciu_sumos = sum(V_fragmentas,2);
    % naikiname tas eilutes, kur suma lygi pločiui
    V_fragmentas(eiluciu_sumos == plotis,:) = [];
    objektai{k}=V_fragmentas;% įrašome vietoje neapkarpyto
end

% vieno iš vaizdo fragmentų atvaizdavimas
figure(6),
for k = 1:Skaicius
   subplot(pvz_eiluciu_sk,raidziu_sk,k), imshow(objektai{k})
end

%% Suvienodiname vaizdo fragmentų dydžius iki 70x50
for k=1:Skaicius
    V_fragmentas=objektai{k};
    V_fragmentas_7050=imresize(V_fragmentas,[70,50]);
    % padalinkime vaizdo fragmentą į 10x10 dydžio dalis
    for m=1:7
        for n=1:5
            % apskaičiuokime kiekvienos dalies vidutinį šviesumą 
            Vid_sviesumas_eilutese=sum(V_fragmentas_7050((m*10-9:m*10),(n*10-9:n*10)));
            Vid_sviesumas((m-1)*5+n)=sum(Vid_sviesumas_eilutese);
        end
    end
    % 10x10 dydžio dalyje maksimali šviesumo galima reikšmė yra 100
    % normuokime šviesumo reikšmes intervale [0, 1]
    Vid_sviesumas = ((100-Vid_sviesumas)/100);

    % rezultatą (požymius) neuronų tinklui patogiau pateikti stulpeliu
    Vid_sviesumas = Vid_sviesumas(:);

    % išsaugome apskaičiuotus požymius į bendrą kintamąjį
    pozymiai{k} = Vid_sviesumas;
end