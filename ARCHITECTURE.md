# Architecture du projet et correspondance avec le diagramme

Ce document décrit le rôle de chaque fichier source en relation avec le diagramme de traitement d'image fourni.

## Vue d'ensemble

Le projet implémente un pipeline de traitement vidéo utilisant GStreamer pour la gestion des flux et du code C++/CUDA pour l'algorithme de détection de mouvement.

## Correspondance Fichiers <-> Diagramme

### 1. Entrée et Gestion du Flux (Frame t)
*   **`src/stream.cpp`** : Point d'entrée de l'application. Il configure le pipeline GStreamer qui capture la vidéo (fichier ou caméra), décode les images ("Frame t") et les envoie au filtre de traitement.
*   **`src/gstfilter.c` / `src/gstfilter.h`** : Plugin GStreamer personnalisé (`myfilter`). Il intercepte chaque "Frame t" du pipeline et appelle la fonction de traitement `cpt_process_frame`.

### 2. Cœur du Traitement (Background Estimation, Difference, Mask, Alert)
Ces fichiers contiennent la logique algorithmique décrite dans le diagramme (estimation de fond, différence, nettoyage du masque, alerte).
*   **`src/Compute.hpp`** : Définit l'interface de traitement (`cpt_init`, `cpt_process_frame`).
*   **`src/Compute.cpp`** : Implémentation CPU des algorithmes :
    *   *Background estimation process* & *Internal BEP state*
    *   *Difference* & *Change mask at t*
    *   *Mask cleaning process*
    *   *Alerting process*
*   **`src/Compute.cu`** : Implémentation GPU (CUDA) des mêmes algorithmes pour l'accélération matérielle.

### 3. Structures de Données et Utilitaires
*   **`src/Image.hpp`** : Classe utilitaire pour manipuler les buffers d'images (pixels, dimensions) utilisés tout au long du pipeline (Frame t, Background image, Masks).
*   **`src/argh.h`** : Bibliothèque pour gérer les arguments de la ligne de commande (choix CPU/GPU, fichier d'entrée).

### 4. Éléments Visuels (Overlay)
Bien que non explicite dans le diagramme algorithmique, ces fichiers servent probablement à l'affichage final (sur l'image de sortie).
*   **`src/logo.c` / `src/logo.h`** : Gestion de l'affichage d'un logo ou d'indicateurs visuels sur l'image traitée.
*   **`src/stb_image.c` / `src/stb_image.h`** : Bibliothèque pour charger des images (comme le logo) depuis le disque.

---

## Résumé du flux de données

1.  **`stream.cpp`** lance GStreamer.
2.  GStreamer extrait une **Frame t**.
3.  **`gstfilter.c`** reçoit la frame et la passe à **`Compute.cpp/cu`**.
4.  **`Compute`** exécute la chaîne : *Background Est. -> Difference -> Mask -> Cleaning -> Alert*.
5.  Le résultat est renvoyé à GStreamer pour affichage ou sauvegarde.
