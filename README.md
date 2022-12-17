# solving-elliptic-equations
### Постановка задачи ###
Рассматриваем задачу Дирихле для эллиптического уравнения

```math
	\begin{align}
		-Lu &= f(x, y), \quad (x, y) \in G\\ 
		u &= \mu(x, y), \quad (x, y) \in \Gamma,
	\end{align}
```
где
```math
	\begin{equation}
		Lu =  \frac{\partial }{\partial x} \Bigl(p(x, y) \frac{\partial u}{\partial x}\Bigr) + \frac{\partial }{\partial y} \Bigl(q(x, y) \frac{\partial u}			{\partial y}\Bigr).
	\end{equation}
```
Здесь $p(x, y)$, $q(x, y)$ - достаточно гладкие функции такие, что 
```math
	\begin{align*}
		0 < c_1 \le p(x, y) \le c_2, \\
		0 < d_1 \le q(x, y) \le d_2,
	\end{align*}
```
где $c_1, c_2, d_1, d_2$ - постоянные.

Обозначим $\bar{G} = G \cup \Gamma = \{0 \le x \le l_x, 0 \le y \le l_y\}$. Разобъем $[0, l_x]$ на $N$ равных частей и обозначим $h_x = l_x / N$. Аналогично поступим с отрезком $[0, l_y]$ и обозначим $h_y = l_y / M$.

Построим сетку узлов $\bar{\omega} = \{(x_i, y_j), \quad 0 \le i \le N, 0 \le j \le M\}$, где $x_i = i h_x$, $y_j = j h_y.$ Будем разделять узлы как на внутренние так и на граничные.

### Разностная аппроксимация задачи Дирихле ###
Обозначим $u_{i,j} = u(x_i, y_j)$. Заменяем оператор $L$ во всех внутренних узлах разностным оператором
```math
	\begin{equation}
		L_hu_{i,j}=p_{i+\frac{1}{2},j} \frac{u_{i+1,j} - u_{i,j}}{h_{x}^2}-p_{i-\frac{1}{2},j} \frac{u_{i,j} - u_{i-1,j}}{h_{x}^2}+q_{i,j+\frac{1}{2}} 			\frac{u_{i,j+1} - u_{i,j}}{h_{y}^2}-p_{i,j-\frac{1}{2}} \frac{u_{i,j} - u_{i,j-1}}{h_{y}^2},
	\end{equation}
```
```math
	\begin{align*}
		p_{i+\frac{1}{2},j} &= p(x_i + h_x/2, y_j), & q_{i,j+\frac{1}{2}} = q(x_i, y_j + h_y/2),\\
		p_{i-\frac{1}{2},j} &= p(x_i - h_x/2, y_j), & q_{i,j-\frac{1}{2}} = q(x_i, y_j - h_y/2),
	\end{align*}
```
$\text{где } 1 \le i \le N-1;  1\le j \le M-1.$

Таким образом, задаче выше ставим в соответствие разностную задачу: найти сеточную функцию, удовлетворяющую во внутренних узлах уравнениям $-L_hu_{i,j} = f_{i,j}$ и принимающую в граничных узлах заданные значения:
```math
	\begin{equation*}
		\begin{cases}
			u_{i,0} &= \mu(x_i, 0),\, 0 \le i \le N;\\
			u_{i,M} &= \mu(x_i, l_y), 0 \le i \le N;\\
			u_{0,j} &= \mu(0, y_j),\, 1 \le j \le M-1;\\
			u_{N,j} &= \mu(l_x, y_j), 1 \le j \le M-1.
		\end{cases}
	\end{equation*}
```

### Реализованные методы ###

1. Метод простой итерации
2. Метод итерации с оптимальным параметром
3. Метод Зейделя (Некрасова)
4. Метод верхней релаксации
5. Метод с Чебышевским набором параметров
6. Попеременно-треугольный итерационный метод
