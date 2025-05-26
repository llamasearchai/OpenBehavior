// MongoDB initialization script

db = db.getSiblingDB('openbehavior');

// Create collections
db.createCollection('evaluations');
db.createCollection('prompts');
db.createCollection('templates');
db.createCollection('experiments');
db.createCollection('users');

// Create indexes for better performance
db.evaluations.createIndex({ "text_id": 1 });
db.evaluations.createIndex({ "timestamp": -1 });
db.evaluations.createIndex({ "evaluation_type": 1 });
db.evaluations.createIndex({ "model": 1 });

db.prompts.createIndex({ "template_id": 1 });
db.prompts.createIndex({ "created_at": -1 });

db.templates.createIndex({ "name": 1 }, { unique: true });
db.templates.createIndex({ "categories": 1 });

db.experiments.createIndex({ "created_at": -1 });
db.experiments.createIndex({ "status": 1 });

// Create admin user
db.users.insertOne({
  username: "admin",
  email: "admin@openbehavior.ai",
  role: "admin",
  created_at: new Date(),
  active: true
});

print("OpenBehavior database initialized successfully");