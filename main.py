# Importation des modules necessaires
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

# La classe Graphe contient les donnees globale du swarm : le maximum globale, le nombre de particule, la matrice d'adjacence
# Il contient aussi la liste des particules qui lui sont attribuees
# Les methodes de la classe sont des outils pour faire fonctionner l'algorithme contenu dans la fonction "colors"
class Graphe :
    
    # Initialisation des attributs du graphe
    def __init__(self,R):
        self.R = R
        self.dim = len(R)
        self.particles = []
        self.tu = False
        self.Pg = False
    
    # Cree N particules de positions et velocites aleatoires dans les bons espaces
    def create_particles(self,N):
        for k in range(N):
            #creation des vecteurs positions et velocites
            v = np.random.random(self.dim)
            x = np.random.randint(4, size=self.dim)
            #ajout d'une particule a la liste de particules du graphe
            self.particles = np.append(self.particles, Particle(x,v))

    # Met a jour l'ensemble des particules selon l'algorithme
    def update_particles(self,w,Pg,c1,c2,fit):
        N = len(self.particles)
        for k in range(N):
            self.particles[k].update(w,Pg,c1,c2,fit)
    
    # Met a jour des attributs du graphe
    def update_graphe(self,t,fit):
        for x in self.particles :
            
            if type(self.Pg) == bool :
                self.Pg = x.pi
            if fit(x.pi)< fit(self.Pg) :
                self.Pg = x.pi
                self.tu = t
    
    # Appelle le reinitialisation des velocites des particules
    def reset_particles(self):
        for x in self.particles :
            x.reset()

    # Fais tourner l'algorithme de coloration
    # Retourne le nombre de conflit de couleur de la solution obtenue, le nombre d'iterations, et le graphique colore.
    def colors(self,M,N):
        # On reinitialise les particules si elles existent :
        self.particles = []
        self.Pg = False
        # On initialise les parametres necessaires
        u = 10 
        # Poids des x.pi, x.Pg, et de la part d'aleatoire dans les mises a jour des particules
        w = 2
        c1 = 2
        c2 = 1.8
        
        #on cree le bon nombre de particules
        self.create_particles(N)
        
        #On definit la fonction fitness, qui compte le nombre de conflits de couleurs pour une solution donnee
        def fit(x):
            conflits = 0
            for i in range(self.dim):
                for j in range(i+1,self.dim):
                    if self.R[i][j] == 1 :
                        if x[i] == x[j] :
                            conflits += 1
            
            
            return(conflits)
        
        # Tant que la solution optimale n'a pas ete trouvee, et que le nombre maximale d'iteration n'a pas ete atteint
        t=0 # Nombre d'iterations
        self.update_graphe(t,fit)
        while fit(self.Pg)!= 0 and t<=M:
            t+=1
            
            # On met a jour les positions des particules, et les attributs du graphe
            self.update_particles(w,self.Pg,c1,c2,fit)
            self.update_graphe(t,fit)
            
            # Si la meilleur solution n'evolue pas, on reinitialise aleatoirement les velocites
            if (t - self.tu > u) :
                self.reset_particles()
        
        # Trace la solution finale obtenue par l'algorithme, en colorant les noeuds dans leur couleur attribuee
        color_map =[]
        for i in range(len(self.Pg)) :
            if self.Pg[i] == 0 :
                color_map.append('red')
            elif self.Pg[i] == 1 :
                color_map.append('blue')
            elif self.Pg[i] == 2 :
                color_map.append('yellow')
            elif self.Pg[i] == 3 :
                color_map.append('green')
        
        Gr = nx.from_numpy_matrix(self.R)
        pos=nx.spring_layout(Gr)
        
        nx.draw(Gr, node_color=color_map, with_labels=True)
        plt.show()
        
        return(fit(self.Pg), t)
# --------------------------------------------------------
# La classe Particle contient les donnees connues par la particule : sa position, sa velocite et sa meilleur performance
# Les methodes de la particule servent a mettre a jour sa position en fonction de sa velocite, et a gerer sa memoire
# Elles sont appelees par la classe Graphe qui detient un certain nombre de particules.
class Particle ():
    # Initialisation des attributs de la particule
    def __init__(self, x, v):
        self.x = x
        self.v = v
        
        self.pi = x
    
    # Mise a jour des coordonnees de la particule selon l'algorithme modified PSO
    def update(self, w, Pg, c1, c2, fit):
        
        # Mise a jour de la velocite
        self.v = w*self.v + c1*np.random.random()*(self.pi-self.x) + c2*np.random.random()*(Pg-self.x)
        
        # Fonction sigmoide
        def S(v):
            return 1/(1+np.exp(-v))
        
        # Fonction aleatoire caracteristique de la quaternary PSO
        def f(v,r):
            resultat = []
            for i in range(len(v)):
                elt = v[i]
                rdm = np.random.random()
                if (rdm>r and rdm<S(elt)):
                    resultat.append(0)
                if (rdm<r and rdm<S(elt)):
                    resultat.append(1)
                if (rdm<=r and rdm>=S(elt)):
                    resultat.append(2)
                if (rdm>=r and rdm>=S(elt)):
                    resultat.append(3)
            return(resultat)
        
        # Mise a jour de la position
        self.x = (self.x + f(self.v,0.5))%4
        
        # Memorisation de la position en cas de meilleur performance
        if (fit(self.pi)>fit(self.x)):
            self.pi = self.x

    # Reinitialisation de la vitesse
    def reset(self):
        n = len(self.v)
        self.v = np.random.random(n)
        
        
if __name__ == '__main__' :        
    # Application de l'algorithme après generation d'un graphe plan aleatoire
    np.random.seed(1)

    n = 20      # Nombre de noeuds du graphe
    M = 10000   # Nombre maximum d'iterations pour l'algorithme
    N = 100     # Nombre de particule utilisees pour l'algorithme

    # Generation d'un graphe aleatoire, jusqu'a ce qu'il soit planaire
    # Remarque : 
    # Aucune methode plus efficace de generation de graphe planaire aleatoire n'a ete trouvee 
    # Pour un petit nombre de noeud (<50) cette methode reste satisfaisante
    test = False
    while not (test) :
        P = np.random.randint(0,2,(n,n)) - np.identity(n)
        P = (P + np.transpose(P))/2
        P[P > 1 ] = 1
        P[P < 1 ] = 0

        G = nx.from_numpy_matrix(P)
        test = nx.check_planarity(G)[0]

    print('Graphe genere')
    print("Lancement de l'algorithme")
    # Recuperation de la matrice d'adjacence du graphe
    P = nx.to_numpy_array(G)

    # Application de l'algorithme
    G = Graphe(P)
    G.colors(M,N)