# Blocked-AllPairsShortestPath

## Todo

* F.W. classico <-- Zuccato
* v. naive paper non parallela in C <-- Zuccato (inizia, se hai problemi parliamone così ci prendo confidenza anche io; pusha spesso così resto aggiornato)

* generazione grafi casuali di prova e mecc. per lettura input <-- Lena

* v. naive con mem. globale
* provare ad usare la shared memory

## 15/07 - Stato della situazione

*   abbiamo implementato Floyd Warshall classico e a blocchi, utilizzando come strutture dati:
    -   la matrice come array di array puntatori (<code>int** m</code>, accedibile con la sintassi <code>m[i][j]</code>)
    -   la matrice vettorizzata come array di puntatori (<code>int* m</code>, accedibile con la sintassi <code>m[i*n+j]</code>)

*   Dalla struttura del Floyd Warshall a blocchi, abbiamo estratto un metodo <code>execute_round(m, n, t, row, col, B)</code>, che dati:
    -   una matrice <code>m</code> di dim. <code>n*n</code>, divisa logicamente in blocchi <code>B*B</code> e rappresentante gli attuali cammini migliori di un grafo G.
    -   un blocco <code>(t,t)</code>
    -   un blocco <code>(row,col)</code>

    confronta per ogni <code>(i,j) in (row,col)</code> il percorso <code>i->j</code> con tutti i percorsi <code>i->k->j</code> con i vari nodi <code>k<code> del blocco <code>(t,t)</code>.

*   Dato questo metodo, si è creata una v. parallela del F.W. a blocchi  implementando:
    -   un codice host di controllo analogo a quello del F.W. a blocchi non parallelo
    -   una versione parallela del metodo <code>execute_round</code> (chiamata <code>execute_round_device_v1_0</code>) che esegue su device e parallelizza con <code>n*n</code> thread i due cicli <code>for</code> più interni di <code>execute_round</code>.Sostanzialmente, ogni thread ha la responsabilità di verificare tutti i percorsi i percorsi <code>i->k->j</code> - con <code>k<code> del blocco <code>(t,t)</code> - per un certa coppia <code>(i,j) in (row,col)</code>.

    In questa prima versione di codice parallelo (<code>floyd_warshall_blocked_device_v1_0</code>) si esegue un alg. di struttura analoga alla versione di F.W. al blocchi e si parallelizza solo le singole esecuzioni di round (*). Si noti che, anche se non sono assolutamente necessari, si lanciano sempre <code>n*n</code> thread, dei quali si attivano però soltanto quelli corrispondenti al blocco <code>(row,col)</code>.

    Nel frattempo, tra un'esecuz. e l'altra si mantiene la matrice nella memoria del device.

*   Abbiamo predisposto i comandi per la compilazione + qualche funzione "di untility" per la gestione delle matrici, dei test e di una futura lettura dell'input.

Linee di sviluppo da esplorare:

*   passare immediatamente ad una rappresentazione della matrice - e dell'indirizzamento dei thread - più consoni, magari esplorando:
    -   <code>cudaMalloc2D</code>
    -   <code>cudaMempitch</code>

*   rimuovere le inefficenze nel lancio dei thread (ha veramente senso lanciare <code>n*n</code> thread se poi ad eseguire sono soltanto <code>B*B</code>?)

*   valutare di parallelizzare meglio le singole fasi di ogni round (*)

*   valutare l'uso della shared memory


