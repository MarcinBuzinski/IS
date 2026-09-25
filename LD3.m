clc
clear all
close all
x = 0.1:1/22:1; 
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;
plot(x, d);
%% =======================
r1 = 0.19;
r2 = 0.21;
c1 = 0.19;
c2 = 0.92;

w1 = rand(1);
w2 = rand(1);
b = rand(1);

eta = 0.1;
%% =======================

for iter=1:100000
    for i=1:length(x)
    y1 = exp(-(x(i)-c1)^2/(2*r1^2));
    y2 = exp(-(x(i)-c2)^2/(2*r2^2));
    
    v = y1 * w1 + y2 * w2 + b;
    y = v;
    
    e = d(i) - y;
    
    w1 = w1 + eta * e * y1;
    w2 = w2 + eta * e * y2;
    b = b + eta * e;
    end
end

%% ====================

x_new =  0.1:1/22:1;
Y = zeros(1, length(x_new));
for i=1:length(x_new)
    y1 = exp(-(x_new(i)-c1)^2/(2*r1^2));
    y2 = exp(-(x_new(i)-c2)^2/(2*r2^2));
    
    v = y1 * w1 + y2 * w2 + b;
    y = v;
    Y(i) = y;
end
hold on
plot(x_new, Y, 'r');