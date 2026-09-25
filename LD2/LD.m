clc
clear all
close all
x = 0.1:1/22:1;
d = ((1 + 0.6*sin(2*pi*x/0.7)) + 0.3*sin(2*pi*x))/2;
plot(x,d);
%% tinklo inicializavimnas
%pirmas sluoksnis
w11_1 = rand(1);
w21_1 = rand(1);
w31_1 = rand(1);
w41_1 = rand(1);
w51_1 = rand(1);

b1_1 = rand(1);
b2_1 = rand(1);
b3_1 = rand(1);
b4_1 = rand(1);
b5_1 = rand(1);

%antras sluoksnis
w11_2 = rand(1);
w12_2 = rand(1);
w13_2 = rand(1);
w14_2 = rand(1);
w15_2 = rand(1);

b1_2 = rand(1);

eta = 0.1;
%%
%feed forward dalis
%tinklo atasaks
for iter=1:100000
for i=1:length(x)
    v1_1= x(i) * w11_1 + b1_1;
    v2_1= x(i) * w21_1 + b2_1;
    v3_1= x(i) * w31_1 + b3_1;
    v4_1= x(i) * w41_1 + b4_1;
    v5_1= x(i) * w51_1 + b5_1;
    %1sluoksnis aktyvacijos funkcija
    y1_1 = 1 / (1 + exp(-v1_1));
    y2_1 = 1 / (1 + exp(-v2_1));
    y3_1 = 1 / (1 + exp(-v3_1));
    y4_1 = 1 / (1 + exp(-v4_1));
    y5_1 = 1 / (1 + exp(-v5_1));
    %2 sluoksnis pasverta suma
    v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + y3_1 * w13_2 + y4_1 * w14_2 + y5_1 * w15_2 + b1_2;
    %isejimo sluoksnio aktyvacija, tiesine funkcija cia - isejimas
    %proporcingas iejimui
    y1_2 = v1_2;
    %tinklo isejimas
    y = y1_2;
    %klaidos skaiciavimas
    e = d(i)-y;
 

    %svoriu atnaujinimas
%gradiento skaiciavimnas
%isejimo sluoksnis
    delta1_2 = e;
   %pasleptasis sluoksnis
   delta1_1=(1 / (1 + exp(-v1_1))) * delta1_2 * w11_2;
   delta2_1=(1 / (1 + exp(-v2_1))) * delta1_2 * w12_2;
   delta3_1=(1 / (1 + exp(-v3_1))) * delta1_2 * w13_2;
   delta4_1=(1 / (1 + exp(-v4_1))) * delta1_2 * w14_2;
   delta5_1=(1 / (1 + exp(-v5_1))) * delta1_2 * w15_2;

   %atnaujinami svoriai
  %atnaujinami atbuline tavrka, pirmas atnaujinamas isejimo sluoksnis

  %isejimo sluoksnio
  w11_2 = w11_2 + eta * delta1_2 * y1_1;
  w12_2 = w12_2 + eta * delta1_2 * y2_1;
  w13_2 = w13_2 + eta * delta1_2 * y3_1; 
  w14_2 = w14_2 + eta * delta1_2 * y4_1; 
  w15_2 = w15_2 + eta * delta1_2 * y5_1; 
  b1_2 = b1_2 + eta * delta1_2;
  %paslepto sluoksnio
  w11_1 = w11_1 + eta * delta1_1 * x(i);
  w21_1 = w21_1 + eta * delta2_1 * x(i);
  w31_1 = w31_1 + eta * delta3_1 * x(i);
  w41_1 = w41_1 + eta * delta4_1 * x(i);
  w51_1 = w51_1 + eta * delta5_1 * x(i);
  b1_1 = b1_1 + eta * delta1_1;
  b2_1 = b2_1 + eta * delta2_1;
  b3_1 = b3_1 + eta * delta3_1;
  b4_1 = b4_1 + eta * delta4_1;
  b5_1 = b5_1 + eta * delta5_1;
end
end
%% -- 
x_new =  0.1:1/22:1;
Y = zeros(1, length(x_new));
for i=1:length(x_new)
    v1_1= x(i) * w11_1 + b1_1;
    v2_1= x(i) * w21_1 + b2_1;
    v3_1= x(i) * w31_1 + b3_1;
    v4_1= x(i) * w41_1 + b4_1;
    v5_1= x(i) * w51_1 + b5_1;
    %1sluoksnis aktyvacijos funkcija
    y1_1 = 1 / (1 + exp(-v1_1));
    y2_1 = 1 / (1 + exp(-v2_1));
    y3_1 = 1 / (1 + exp(-v3_1));
    y4_1 = 1 / (1 + exp(-v4_1));
    y5_1 = 1 / (1 + exp(-v5_1));
    %2 sluoksnis pasverta suma
    v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + y3_1 * w13_2 + y4_1 * w14_2 + y5_1 * w15_2 + b1_2;
    %isejimo sluoksnio aktyvacija, tiesione funkcija cia - isejimas
    %proporcingas iejimui
    y1_2=v1_2;
    %tinklo isejimas
    y=y1_2;
    e=d(i)-y;
    Y(i) = y;
end
hold on
plot(x_new, Y, 'r');
