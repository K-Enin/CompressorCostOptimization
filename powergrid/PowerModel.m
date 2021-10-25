function [F1,F2] = PowerModel(x)

G = [0 0 0 0 0 0 0 0 0; 
    0 0 0 0 0 0 0 0 0; 
    0 0 0 0 0 0 0 0 0; 
    0 0 0 3.3074 -1.9422 0 0 0 -1.3652;
    0 0 0 -1.9422 3.2242 -1.282 0 0 0;
    0 0 0 0 -1.282 2.4371 -1.1551 0 0;
    0 0 0 0 0 -1.1551 2.7722 -1.6171 0;
    0 0 0 0 0 0 -1.6171 2.8047 -1.1876;
    0 0 0 -1.3652 0 0 0 -1.1876 2.5528];

B = [-17.3611 0 0 17.3611 0 0 0 0 0;
    0 -16 0 0 0 0 0 16 0;
    0 0 -17.0648 0 0 17.0648 0 0 0;
    17.3611 0 0 -39.3089 10.5107 0 0 0 11.6041;
    0 0 0 10.5107 -15.8409 5.5882 0 0 0;
    0 0 17.0648 0 5.5882 -32.1539 9.7843 0 0;
    0 0 0 0 0 9.7843 -23.3032 13.698 0;
    0 16 0 0 0 0 13.698 -35.4456 5.9751;
    0 0 0 11.6041 0 0 0 5.9751 -17.3382];

N = 9;
F1 = zeros(9,1);
F2 = zeros(9,1);

global P_at_t Q_at_t;

P = [x(1), 1.63, 0.85,0, P_at_t, 0, -1,0,-1.25];
Q = [x(2), x(3), x(4), 0, Q_at_t, 0, -0.35, 0, -0.5];
V = [1,1,1,x(5),x(6),x(7),x(8),x(9),x(10)];
phi = [0,x(11),x(12),x(13),x(14),x(15),x(16),x(17),x(18)];

for k = 1:N
    for j = 1:N
        F1(k) = F1(k) + abs(V(k))*abs(V(j))*(G(k,j)*cos(phi(k)-phi(j)) + B(k,j)*sin(phi(k)-phi(j)));
        F2(k) = F2(k) + abs(V(k))*abs(V(j))*(G(k,j)*sin(phi(k)-phi(j)) - B(k,j)*cos(phi(k)-phi(j)));
    end
    F1(k) = F1(k) - P(k);
    F2(k) = F2(k) - Q(k);
end 