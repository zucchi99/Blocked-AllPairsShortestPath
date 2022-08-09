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
    -   una versione parallela del metodo <code>execute_round</code> (chiamata <code>execute_round_device_v1_0</code>) che esegue su device e parallelizza con <code>n*n</code> thread i due cicli <code>for</code> più interni di <code>execute_round</code>.Sostanzialmente, ogni thread ha la responsabilità di verificare tutti i percorsi i percorsi <code>i->k->j</code> - con <code>k</code> del blocco <code>(t,t)</code> - per un certa coppia <code>(i,j) in (row,col)</code>.

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

## 19/07 - Stato della situazione

Oggi parallelamente si sono sperimentate:

*   la parallelizzazione delle fasi 2 e 3 dell'algoritmo <code>floyd_warshall_blocked_device_v1_1</code>
*   l'utilizzo di differenti strutture dati per eseguire l'algoritmo <code>floyd_warshall_blocked_device_v1_1</code>

### Parallelizzazione delle fasi

Si è scritta una nuova versione <code>floyd_warshall_blocked_device_v1_2</code>, che a differenza della precedente fa uso di 3 funzioni device differenti:

<code>execute_round_device_v1_2_phase_1</code> è molto simile a <code>execute_round_device_v1_0</code>, si lancia sui <code>B*B</code> thread del blocco self dipendente della fase 1 dei round.

<code>execute_round_device_v1_2_phase_2</code> e <code>execute_round_device_v1_2_phase_3</code> sono due nuove funzioni che svolgono rispettivamente la fase 2 e la fase 3 dei round. Esse:

*   si lanciano su una griglia completa di <code>n*n</code> blocchi 
*   i thread sono sempre indicizzati a griglia, ma questa volta <code>tid_x</code> e <code>tid_y</code> rappresentano gli indici assoluti della cella della matrice sulla quale il blocco deve lavorare
*   sono divisi in <code>(B/n)^2</code> blocchi CUDA
*   hanno sempre come struttura principale un ciclo che itera su tutti i nodi del blocco <code>t</code>
*   attivano il confronto SSE il thread corrisponde ad uno dei blocchi della fase corrispondente (attraverso una serie di espressioni booleane che verificano il posizionamento del blocco). Si noti che quest'ultima caratteristica potrebbe essere ragione di inefficienza, in quanto si lanciano sempre più blocchi di quanti effettivamente se ne usano, in particolare per la fase 2.


Si noti che il codice del confronto effettuato da queste funzioni è molto simile a quanto già presente.

Si noti anche che per tutte e tre queste funzioni è importantissimo tenere un <code>__syncthread()</code> al termine di ogni iterazione del ciclo su tutti i nodi del blocco <code>t</code>.

### Scelta della struttura dati

TODO: write log

### Sviluppi per i prossimi giorni

*   misurare l'efficienza di <code>floyd_warshall_blocked_device_v1_2</code>
*   confrontare le varie soluzioni sviluppate con le diverse strutture dati e scegliere quale mantenere per gli sviluppi futuri
*   creare una versione pulita del codice <code>floyd_warshall_blocked_device_v1_*</code>, da ottimizzare poi al massimo con la memoria e da tenere come riferimento quando si svilupperanno le versioni più avanzate.

## 20/07 - Stato della situazione

Si è creata una versione <code>floyd_warshall_blocked_device_v1_3</code> simile alla precedente, che però questa volta ottimizza il numero di blocchi lanciati per le fasi 2 e 3. 

*   Per la fase 2 si lanciano <code>2*(n/B-1)</code> blocchi, aventi la seguente struttura (rispetto alla griglia)

    <code>
        //  L1  L2  L3  R1  R2
        //  U1  U2  U3  D1  D2
        
        //  .   .   .   U1  .   .
        //  .   .   .   U2  .   .
        //  .   .   .   U3  .   .
        //  L1  L2  L3  -   R1  R2
        //  .   .   .   D1  .   .
        //  .   .   .   D2  .   .
    </code>

*   Per la fase 3 si lanciano <code>(n/B-1)*(n/B-1)</code> blocchi, aventi la seguente struttura (rispetto alla griglia)

    <code>
        //  UL  UL  UL  UR  UR
        //  UL  UL  UL  UR  UR
        //  UL  UL  UL  UR  UR
        //  DL  DL  DL  DR  DR
        //  DL  DL  DL  DR  DR

        //  UL  UL  UL  -   UR  UR
        //  UL  UL  UL  -   UR  UR
        //  UL  UL  UL  -   UR  UR  
        //  -   -   -   -   -   - 
        //  DL  DL  DL  -   DR  DR
        //  DL  DL  DL  -   DR  DR
    </code>

*   (Per la fase 1 si lancia sempre solo un blocco)

Per fare ciò si sono ri-scritte le procedure <code>execute_round_device_v1_3_phase_2</code> e <code>execute_round_device_v1_3_phase_3</code>; <code>execute_round_device_v1_2_phase_1</code> invece è rimasta invariata.

La razionalizzazione del numero di blocchi lanciati si paga con una complicazione del metodo di indirizzamento, che diventa più complicato (specie per la fase 2).

## 01/08 - Stato della situazione

### Sperimentazione della struttura dati pitch

Parallelamente a quanto svolto per l'implementazione delle versioni 1.1-1.3, si ha effettuato anche una sperimentazione sulla struttura dati "pitched". Tale struttura dati consiste in una modalità particolare di allocazione della memoria di cuda (in particolare, per l'allocazione di matrici bi-dimensionali) pensata per prevenire il problema del bank conflict. 

In pratica, il "pitch" consiste nell'allocazione di una struttura dati con "un padding" al termine di ogni riga, utile appunto a prevenire il bank conflict in caso di accesso per colonna.

Si è sperimentata la possibilità di allocare e usare spazi pitched, portando alla creazione delle due nuove versioni <code>floyd_warshall_blocked_device_v1_1_pitch</code> e <code>floyd_warshall_blocked_device_v1_3_pitch</code> (che ricalcano come struttura dell'algoritmo le loro corrispondenti normali, ma allocano la memoria in modo pitched e adeguano di conseguenza le tecniche di accesso).

A seguito di un test, tali versioni non si sono dimostrate più efficienti delle loro corrispondenti. Ciò è dovuto al fatto che si sta utilizzando la memoria globale e non la shared.

Probabilmente, neache usando la shared si otterranno risultati tanto migliori (visto che comunque si accede contemporaneamente a tutte le celle della matrice e il conflitto di banco in tanti casi probabilmente sarà inevitabile).

### Razionalizzazione della fase 2 della versione 1.3

Si è creata una nuova versione <code>floyd_warshall_blocked_device_v1_4</code> che segue la stessa struttura della 1.3, ma spezza in due la funzione device della fase 2. Invece di avere una funzione unica che lavora su righe e colonne, si è preferito definire due funzioni distinte, in modo da evitare la prolificazione di strutture decisionali <code>if-else</code> per l'indicizzazione.

## 02/08 - Stato della situazione

### Refactoring del Make, dell'org. dei file e dello statistical test

Per agevolare il lavoro:
 
* è stato riorganizzato il <code>Makefile</code> per fare modo di ri-usare lo stesso comando di compilazione per compilare diversi file (facendo uso di una variabile);

* anche la struttura dei file è stata ri-organizzata per agevolare la gestione delle diverse versioni;

* si è ri-scritto il codice del test statistico, in modo tale da poter effetturare test casuali con input di dimensione e blocking factor che non siano potenze di due

### Versione 2.0 e uso della shared memory

A seguito di numerosi sforzi, si è implementata la prima versione del programma che fa uso di shared memory (<code>floyd_warshall_blocked_device_v2_0</code>). Tale versione:

*    mantiene la stessa struttura di <code>floyd_warshall_blocked_device_v1_4</code>

*   in fase 1, fa uso di uno spazio condiviso nel blocco di dimensione <code>B*B</code> per memorizzare temporaneamente il blocco self-dependend <code>(t,t)</code>. Prima di terminare l'esecuzione su device, si copiano i risultati sulla memoria globale.

*   nelle due funzioni della fase 1 e della fase 2, fa uso di uno spazio shared dinamico di dimensione <code>B*B*2</code> per memorizzare
    -   il blocco self-dependent <code>(t,t)</code>
    -   il blocco della riga/della colonna sul quale si sta lavorando.

    Come per la fase 1, prima di terminare l'esecuzione su device, si copiano i risultati sulla memoria globale (ovviamente, si copia solo il blocco della riga/colonna attiva, non il blocco <code>(t,t)</code>).

*   Per quanto riguarda la fase 3, si utilizza sempre uno spazio condiviso di dimensione <code>B*B*2</code>, dove si conservano i due blocchi corrispondenti alla "proiezione" del blocco corrente sulla riga e sulla colonna di <code>(t,t)</code>. 

    Cioè, se si stà lavorando sul blocco <code>(row,col)</code>, allora si copiano <code>(t,col)</code> e <code>(row,t)</code>. Non si memorizza invece nello spazio condiviso il blocco <code>(row,col)</code> in quanto in questa fase ogni thread necessita di accedere soltanto alla sua cella corrispondente, pertanto non avrebbe senso creare uno spazio condiviso; si è preferito quindi usare una variabile locale (e qunidi un resistro dello streaming multi-processor) per memorizzare la singola cella necessaria al thread.

    Al termine della fase 3, ogni thread copia il valore della sua cella corrispondente nella memoria globale.

*   Si noti che in tutto ciò non si avranno mai conflitti in scrittura, in quanto in ogni fase ogni thread scrive solo sulla cella a lui corrispondente.


## 04/08 - Stato della situazione

### Refactoring e pulizia

Prima di procedere, si è fatta un po' di pulizia sulla versione 2.0. Prevalentemente si è fatta un po' di rimozione di prametri e variabili temporanee inutili nelle funzioni delle fasi (cercando di sfruttare meglio le coordinate di thread e di blocco di CUDA). In particolare, si sono rimossi:

* indici inutili
* calcoli inutili
* il parametro B (è sufficiente la dimensione di blocco CUDA)


### Versione 2.1f - Parallelizzazione della prima fase

Successivamente al refactoring, si ha sperimentato lo svolgimento in parallelo di tutti i blocchi self-dependent <code>(t,t)</code> in modo indipendente (<code>floyd_warshall_blocked_device_v2_1f</code>). In pratica, invece di lanciare la fase uno all'inizio di ogni round, si lancia un unico kernel con <code>n/B</code> blocchi (ciascuno che svolge la fase uno su un blocco della shared).

A seguito di un test statistico basato su set di 500 esecuzioni (di diverse dimensioni e con diversi blocking factor), la funzione risulta però non funzionare. Questa è una dimostrazione che i blocchi self-dipendenti - per qualche motivo non ancora noto - non possono essere eseguiti concorrentemente prima del resto del programma.




