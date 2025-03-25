---
title: emoj.yt (emoji URL-shortener)
date: 2020-08-19
excerpt_separator: <!--more-->
tags:
  - code
  - javascript
  - web-development
---

This is a little write up of a very small project I did, inspired by [Coding Garden with CJ](https://coding.garden) on youtube & twitch _(specifically [this video](https://www.youtube.com/watch?v=gq5yubc1u18))_, and [Net Ninja](https://www.thenetninja.co.uk) express tutorials:  
A URL-shortener that uses a sequence of emojis to encode each URL.  
The code is available on [github](https://github.com/lucblassel/), and you can try it out at [emoj.yt](https://emoji-shortener-frontend.vercel.app).

<!--more-->

## What is a URL shortener ?

URLs can be long and a drag to remember or share, thus: URL shorteners _(like [bit.ly](https://bitly.com) for example)_. They work by assigning a random string value (a key) to an URL and then redirecting you to the original URL when accessing this key on their website.  
Let's say I want to share the following URL: _`https://lucblassel.com/blog/implementing-linear-regression`_. If bit.ly assigns it the value _`jFDSijfs`_, then the URL _`bit.ly/jFDSijfs`_ will redirect me to the one I want. Another useful feature si that you can choose the key, so I could use this link _`bit.ly/AwesomeSite`_ as the shortened URL.  
I recently learned that you can have emojis in URLs now thanks to [punycode](https://en.wikipedia.org/wiki/Punycode) and wanted to try it out by implementing a URL shortener that used strings of emojis as keys. So for example I could use _`emoj.yt/👍😂🎉`_ as the shortened URL.

## What did I use in this project ?

I didn't set out to do it this way, but I ended up building a [MERN](https://www.geeksforgeeks.org/mern-stack/) app, meaning an app using [**M**ongoDB](https://www.mongodb.com) as a database, [**E**xpress](http://expressjs.com) as the backend, [**R**eact](https://reactjs.org) as the frontend and [**N**ode.js](https://nodejs.org/en/) as the server.  
This mainly came about when I discovered I was eligible for the [GitHub Student Developer Pack](https://education.github.com/pack) with a free account on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) where I could host my database. I decided then to build the backend with express and host it on [Heroku](https://www.heroku.com), and host the React frontend on [Vercel](https://vercel.com/).

## General structure

For this project I need a couple things:

- A database that can stores records of corresponding URLs and emoji keys _(MongoDB)_.
- A REST API that can enable CRUD operations on that database _(Express)_:
  - a `GET` route to get the URL corresponding to a specific string of emojis
  - a `POST` route that creates a new URL / emoji key record in the database
- A frontend webpage that can submit `GET` and `POST` requests to the API _(React)_.

## The Backend

_The code is available on [github](https://github.com/lucblassel/emoji-shortener-backend)_

The first thing I did was set up my database on Atlas and saved the URL to this database as an env variable on my API server.
To be able to communicate with my MongoDB instance I used [monk](https://automattic.github.io/monk/):

```javascript
const monk = require("monk");

// reading database address + secret
require("dotenv").config();

// connect to database
const db = monk(process.env.MONGODB_URL);
// check connection is established
db.then(() => {
  console.log("Monk connected to database 🔗");
});
// choose collection and create index
const urls = db.get("urls");
urls.createIndex({ emojis: 1 }, { unique: true });
```

Once this was done I could create my express app with some middleware _(cors for dealing with CORS, helmet for dealing with headers, and morgan to log requests)_:

```javascript
const express = require("express");
const morgan = require("morgan");
const cors = require("cors");
const helmet = require("helmet");

// get port
const port = process.env.PORT || 4000;
// get domain name
const domain = process.env.DOMAIN || "localhost";

const app = express();

// set up middleware
app.use(morgan("combined")); // log requests
app.use(cors()); // enable CORS
app.use(helmet()); // secure app & set headers
app.use(express.json()); // parse request body as JSON
```

With the express app initialized and all middlewared up it's time to create some routes. I went with a general `GET` route that returned the last `n` entries in the database, a `GET` ROUTE that allows us to get the URL for a specific emoji string and a `POST` route to allow us to create new records. _(I also made the base GET route return the whole database)_

```javascript
const router = express.Router();

// get all urls from most recent to oldest
router.get("/", async (req, res) => {
  let records = await urls.find({});
  res.json(records.reverse());
});

// get last n urls from most recent to oldest
router.get("/last/:num", async (req, res) => {
  let index = Number(req.params.num);
  let records = await urls.find({});
  res.json(records.slice(-index).reverse());
});

// get specific URL
router.get("/:id", async (req, res, next) => {
  //
  //  Code for specific GET route here
  //
});

// insert new URL
router.post("/newURL", async (req, res, next) => {
  //
  //  Code for POST route here
  //
});
```

For the `/:id` route, the code is quite simple, I just look for the record with the given ID; if it exists I send the JSON version of the record, if not I send an error code:

```javascript
// get specific URL
router.get("/:id", async (req, res, next) => {
  try {
    const url = await urls.findOne({ emojis: req.params.id });
    if (url) {
      res.json({
        emojiURL: `https://${domain}/${url.raw}`,
        redirectURL: url.url,
      });
    } else {
      console.log("not found");
      return res.status(404).send("not found");
    }
  } catch (error) {
    console.log(error);
    return res.status(500).send("Error");
  }
});
```

For the post route, it's slightly more complicated. There are several components I need:

- I need to validate user-input values before inserting them in my database
- if the emoji string is not specified I need to generate a random emoji string
- if an emoji string is specified I need to check it's not already a key in my database.

To check if keys exist and generate random emoji strings I wrote helper functions. In my database I don't use emojis directly as the index, I use the corresponding encoded punycode string. To generate a random sequence of emojis I used [node-emoji](https://github.com/omnidan/node-emoji)

```javascript
const punycode = require("punycode");
const nodeEmoji = require("node-emoji");

// check if key exists
function keyExists(key) {
  urls.findOne({ emojis: key }).then((doc) => {
    if (doc) {
      return true;
    } else {
      return false;
    }
  });
}

// generate random key not in use
function generateRandomEmojis() {
  let emojis;
  let exists;
  do {
    emojis = "";
    for (let i = 0; i < 5; i++) {
      emojis += nodeEmoji.random().emoji;
    }
    exists = keyExists(punycode.encode(emojis));
  } while (exists);
  return emojis;
}
```

To validate the `POST` request parameters before inserting into my database I used [yup](https://github.com/jquense/yup), to make sure the URL and emoji string are valid.

```javascript
// data validation schema
const recordSchema = yup.object().shape({
  emojis: yup
    .string()
    .trim()
    .matches(/^[\w\-]/i), // if not specified we generate one at random
  url: yup.string().trim().url().required(),
});
```

With all this in place I can fill out my `POST` route, with error handling and all the hoopla:

```javascript
// insert new URL
router.post("/newURL", async (req, res, next) => {
  let { emojis, url } = req.body;
  let encodedEmojis = emojis ? punycode.encode(emojis) : undefined;
  try {
    await recordSchema.validate({
      encodedEmojis,
      url,
    });

    if (!encodedEmojis) {
      emojis = generateRandomEmojis();
      encodedEmojis = punycode.encode(emojis);
    } else {
      if (keyExists(encodedEmojis)) {
        throw new Error("This emoji slug already exists... 😿");
      }
    }

    if (encodedEmojis.slice(0, -1) === emojis) {
      throw new Error("There must be at least 1 emoji in the slug... 👹");
    }

    let newURL = { emojis: encodedEmojis, url: url, raw: emojis };
    let created = await urls.insert(newURL);
    console.log(created);
    res.send({ port: port, domain: domain, ...created });
  } catch (error) {
    console.log("Error caught on this req");
    console.log("params", req.params, "body", req.body);
    next(error);
  }
});
```

From there that last thing to do is register the routes and start the API up:

```javascript
// register routes
app.use("/api", router);

// start listening
app.listen(port, () => {
  console.log(`Listening on ${domain}:${port} 🦻`);
});
```

With this I had a functional REST API where I could create shortened URLs and retrieve the original one. After some testing with [Insomnia](https://insomnia.rest) I deployed this API to a Heroku dyno and made sure I still could make requests to the API _(This was my first encounter with CORS nonsense...)_ .  
Now that everything was up and running it was time to do the frontend.

## The Frontend

_The code is available on [github](https://github.com/lucblassel/emoji-shortener-frontend)_

I must say that I do not have a lot of experience with web development so I decided to use React to try and familiarize myself with the framework a bit. So After having bootstraped a little started page with [create-react-app](https://create-react-app.dev) I got started with the tweaking. I am not going to spend a lot of time on this part as is it not what interests me the most but I'll go through the main stuff.  
I decided to have 2 pages, the main page with the latest entries in the database as well as a form that allows the user to shorten their own URL, and a second page when you access a shortened URL. I chose to make this second page instead of redirecting directly to the original URL because I wanted to minimize the potential for harm of this app, and limit people being redirected to malicious URLs directly. Plus having 2 pages meant I could learn how to use the `react-router` with dynamic routes (since technically the second page needs to be generated for every shortened URL in the database).

For all the calls to my backend I used the `fetch` API, and for the user input form I used [Formik](https://formik.org) which was really easy to learn and has top-notch documentation.

I deployed the frontend to Vercel _(where this website is also hosted 😉)_ and connected my domain to it. And voila! it's available at [emoj.yt](https://emoji-shortener-frontend.vercel.app) for all to try.

## In the end

All in all I think this was an interesting dip into the full-stack world, this was the first time I'd written an API and actually went further than a tutorial with React. This allowed me to learn about the fabulous world of HTTP requests as well so that's good _(I guess...)_.  
All in all I'm pretty happy with the result (considering it's my first actual website), and I encourage you to give me feedback if you want, here or on GithHub.
