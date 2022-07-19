# Blocked-AllPairsShortestPath

## Note varie

Comando per compilare tutto:    <code>make fwa fwm fwa_dev read_matrix</code>

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

    confronta per ogni <code>(i,j) in (row,col)</code> il percorso <code>i->j</code> con tutti i percorsi <code>i->k->j</code> con i vari nodi <code>k</code> del blocco <code>(t,t)</code>.

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

## 18/07 - Stato della situazione

A seguito dell'esecuzione di test statistici un po' più seri, abbiamo scoperto che <code>floyd_warshall_blocked_device_v1_0</code> non funziona per input elevati (il risultato ottenuto è differente rispetto a quello ottenuto con il classico <code>floyd_warshall</code> di base eseguito su host).

Abbiamo quindi implementato una nuova sotto-versione <code>floyd_warshall_blocked_device_v1_1</code> che, a differenza della versione base <code>v1_0</code>, fa uso di una nuova funzione device <code>execute_round_device_v1_0</code> che, a differenza della precedente:

*   viene lanciata su <code>B*B</code> thread (solo quelli del blocco da analizzare) invece che sull'intera griglia
*   assume che un blocco CUDA abbia dimensione <code>B*B</code>, che non può essere mai superiore alla massima dimensione del blocco nel device
*   viene lanciata come griglia 2D invece che come array di thread, permettendo un'indicizzazione relativa interna al blocco più semplice (non c'è più solo un <code>tid</code>, bensì un <code>tid_x</code> che rappr. la riga relativa e un <code>tid_y</code> che rappresenta la colonna relativa all'interno del blocco)

All'atto pratico, allo stato attuale si esegue sempre un blocco CUDA singolo per ogni chiamata. Successivamente proveremo a parallelizzare, in più blocchi cuda, le varie esecuzioni delle fasi 2 e 3.


