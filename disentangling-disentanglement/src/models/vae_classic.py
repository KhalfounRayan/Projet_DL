# Un **VAE classique** est un cas particulier du modèle Alpha-Beta VAE. 
# Il peut être obtenu en fixant simplement les paramètres suivants :  

# - **Beta = 1** : pondération standard du terme de régularisation KL-Divergence.  
# - **Alpha = 0** : aucun terme de régularisation supplémentaire.  
# - **Gamma = 0** : aucun ajustement du terme de reconstruction.

# Ainsi, il n'est pas nécessaire de créer une classe dédiée pour le VAE classique. 
# En utilisant la même implémentation et en réglant ces paramètres, on obtient directement un VAE classique. 
# Cela simplifie le code et permet de comparer facilement les résultats avec des variantes comme l'Alpha-Beta VAE.