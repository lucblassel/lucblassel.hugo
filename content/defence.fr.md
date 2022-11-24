---
title: "Soutenance de Thèse  🧑‍🎓"
comments: false
showBreadcrumbs: false
showReadingTime: false
showTOC: false
---

Bonjour à tous,  
Vous le savez sans-doutes, j'ai passé ces dernières années à préparer ma thèse de doctorat en bioinformatique à l'Institut Pasteur. Je soutiendrais donc bientôt cette thèse intitulée **"Des séquences au savoir, améliorer et apprendre des alignements de séquences"**. La soutenance se déroulera **en Anglais**.  

Cette soutenance se déroulera le **2 Décembre** à **13h30**, sur le [campus Jussieu](https://goo.gl/maps/fV5BX99xdRsy6wNy7) de Sorbonne Université. Elle se tiendra dans l'amphithéâtre **Durand** du batiment **Esclangon** *(cf. carte ci-dessous)*. Un évenement de calendrier est disponible [ici](/files/defence.ics).  
Si vous ne pouvez pas venir, pas de soucis! La soutenance sera aussi joignable en visioconférence via Microsoft Teams (lien [ici](https://teams.microsoft.com/l/meetup-join/19%3ameeting_YjQ2NDliYjktMzQ5OS00MDc5LWIyYzItNmFkNjdhMmRmMGM3%40thread.v2/0?context=%7b%22Tid%22%3a%22096815dc-d9eb-4bc3-a5a3-53c77e7d34e2%22%2c%22Oid%22%3a%22efed87df-6c6d-40e8-8d70-14ad4e775309%22%7d))  

La soutenance sera suivie d'un petit pot avec de la nourriture et des boissons dans **l'Atrium** *(cf. carte ci dessous)*, puis d'une soirée dans un bar proche:**Le Baker Street Pub** *(details [ci-dessous](#la-soirée))*. Si vous ne pouvez pas assister à la soutenance ou au pot j'espère que vous pourrez venir au bar!

![Carte du campus](/images/amphi_durand.jpg#center)

----

### Afin que je puisse m'organiser, veuillez remplir [ce formulaire](https://framaforms.org/soutenance-de-luc-lucs-defence-1667484172), également disponible [en bas de page](#formulaire)

## Jury
La soutenance se fera en présence de mon Jury de thèse dont la composition est donnée ci-dessous:

{{< markdowntable >}}

|                                                                           |                     |                    |
| ------------------------------------------------------------------------- | ------------------- | ------------------ |
| [Brona Brejova](http://compbio.fmph.uniba.sk/~bbrejova/)                  | Associate Professor | Rapportrice        |
| [Macha Nikolski](https://dept-info.labri.fr/~macha/)                      | Group Leader        | Rapportrice        |
| [Élodie Laine](http://www.lcqb.upmc.fr/laine/Home.html)                   | Associate Professor | Examinatrice       |
| [Olivier Gascuel](https://isyeb.mnhn.fr/fr/annuaire/olivier-gascuel-7496) | Research Director   | Examinateur        |
| [Jean-Philippe Vert](https://members.cbio.mines-paristech.fr/~jvert/)     | Research Director   | Examinateur        |
| [Paul Medvedev](https://medvedevgroup.com/principal-investigator/)        | Associate Professor | Membre Invité      |
| [Rayan Chikhi](http://rayan.chikhi.name/)                                 | Group Leader        | Directeur de thèse |

{{< /markdowntable >}}

## Déroulement de la soutenance

L'emploi du temps provisoire est le suivant: 

{{< markdowntable >}}

|   |   |   |
|---|---|---|
| **13:30** | **14:15**             | Présentation          |
| **14:15** | **15:00** ~ **15:30** | Questions du Jury     |
| **15:30** | **16:00**             | Déliberations du Jury |
| **16:00** | **16:15**             | Verdict du Jury       |
| **16:15** | **18:00**             | Pot de thèse          |
| **18:00** | **?**                 | Soirée 🎉             |

{{< /markdowntable >}}

## La soirée
La soirée se tiendra au **Baker Street Pub** au *9 rue des boulangers*. Le bar est proche de l'université. On s'y rendra probablement aux alentours de 18h. J'ai réservé un espace semi-privé où nous pourrons mettre notre propre musique. Cependant, cet espace est en sous-sol donc l'accès à internet est très mauvais. Voici donc [un lien](https://open.spotify.com/playlist/1vVTYDkc3iacBF9dWZdGZY?si=c3c5c2b5a7ca4fb1&pt=a8e7de3f76ce9ab1b003c7471a40160f) vers une playlist partagée où vous pouvez ajouter les morceaux que vous voulez:  

{{< rawhtml >}}
<iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/1vVTYDkc3iacBF9dWZdGZY?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
{{< /rawhtml >}}

## Ma thèse

Si vous souhaitez lire le manuscrit il est disponible *(en anglais)* sous forme de site web ([thesis.lucblassel.com](https://thesis.lucblassel.com)) ou alors d'un [document pdf](https://thesis.lucblassel.com/_main.pdf). Bonne chance!

### Résumé 

Dans cette thèse nous étudierons deux problèmes importants en bioinformatique, le premier concernant l’analyse primaire de données de séquencage, et le second concernant l’analyse secondaire de séquence par apprentissage automatique en vue d’obtenir des connaissances biologiques. L’alignement de séquences est l’un des outils les plus puissants et les plus importants dans le domaine de la biologie computationnelle. L’alignement de lectures de séquencage est souvent la première étape de nombreuses analyses telles que la détection de variations de structure, ou l’assemblage de génomes. Les technologies de séquençage à longue lectures ont amélioré la qualité des résultats pour toutes ces analyses. Elles sont, cependant, riches en erreurs de séquençage et posent des problèms algorithmiques à l’alignement. Une technique répandue pour réduire les effets néfastes de ces erreurs est la compression d’homopolymères. Cette technique cible le type d’erreur de séquençage à longue lectures le plus fréquent. Nous présentons une technique plus générale que la compression d’homopolymères, que nous appelons les “mapping-friendly sequence reductions” (MSR). Nous montrons ensuite que certaines de ces MSRs améliorent la précision des alignements de lecture sur des génomes entiers d’humain, de *drosophile* et d’*E. coli*. L’amélioration des méthodes d’alignment de séquences est cruciale pour les analyses en aval. Par exemple, les alignements de séquences multiples sont indispensables pour étudier la pharmaco-résistance des virus. Grâce à la quantité toujours croissante d’alignements de séquences multiples annotés et de haute qualité, il est aujourd’hui devenu possible et utile d’étudier la résistance des virus à l’aide de méthodes d’apprentissage automatique. Nous avons utilisé un très grand alignement de séquences multiples de séquences de VIH britanniques et entraîné plusieurs classificateurs pour distinguer les séquences non-traitées des séquences traitées. En étudiant les variables importantes aux classificateurs, nous identifions des mutations associées à la résistance. Nous avons ensuite supprimé des données, avant l’entraînement, le signal de pharmaco-résistance connu. Nous conservons le pouvoir discriminant des classificateurs, et avons identifié 6 nouvelles mutations associées à la résistance. Une étude plus approfondie a montré que celles-ci étaient très probablement accessoires et liées à des mutations de résistance connues.

## Formulaire

{{< rawhtml >}}
<iframe src="https://framaforms.org/soutenance-de-luc-lucs-defence-1667484172" width="100%" height="800" border="0">
</iframe>
{{< /rawhtml >}}