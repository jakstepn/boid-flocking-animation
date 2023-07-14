Boid Flocking CUDA + OpenGL

Należy dodać GLFW oraz GLAD do folderu zależności.

Możliwa jest zmiana liczby wyświetlanych ryb oraz przeprowadzania obliczeń za pomocą CPU (domyślnie wybrane jest GPU).
Użycie z plikiem wykonywalnym o nazwie "ShoalOfFish":

./ShoalOfFish [liczba rybek, domyślnie 10240] [-c]
Wszystkie parametry są opcjonalne.

-c - aktywowanie CPU do obliczeń.

W Visual Studio opcje te można zmienić w:
Debug -> Properties -> Debugging -> Command Arguments -> wpisać np. "1000 -c"

Aplikację należy uruchomić w trybie Release, bez debuggera w celu otrzymania maksymalnej wydajności.

Po uruchomieniu aplikacji na ekranie wyświetla się okno z animacją. Stworzone okno posiada możliwość zmiany rozmiaru poprzez 
przesunięcie krawędzi okna, lub jego maksymalizację.

W nazwie okna wyświetlana jest kolejno:
FPS: liczba wyświetlanych klatek na sekundę.
Align: wartość współczynnika "Alignment" w algorytmie.
Cohes: wartość współczynnika "Cohesion" w algorytmie.
Separ: wartość współczynnika "Separation" w algorytmie.
Aquar: wartość współczynnika unikania krawędzi akwarium w algorytmie.

Klawisze:
Esc - zamknięcie okna i zatrzymanie aplikacji.
Space - zatrzymanie/wznowienie animacji.

Q - zwiększenie wagi "Alignment" w algorytmie o 0.01.
A - zmniejszenie wagi "Alignment" w algorytmie o 0.01.

W - zwiększenie wagi "Cohesion" w algorytmie o 0.01.
S - zmniejszenie wagi "Cohesion" w algorytmie o 0.01.

E - zwiększenie wagi "Separation" w algorytmie o 0.01.
D - zmniejszenie wagi "Separation" w algorytmie o 0.01.

R - zwiększenie wagi unikania krawędzi akwarium w algorytmie o 0.01.
F - zmniejszenie wagi unikania krawędzi akwarium w algorytmie o 0.01.