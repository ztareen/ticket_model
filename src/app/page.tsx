// src/pages/index.tsx
export default function Home() {
  return (
    <div className="min-h-screen bg-white text-black px-6 py-10">
    <script src="./my-script.js"></script>
      {/* Header */}
      <header className="flex items-center justify-between border-b pb-4 mb-10">
        <h1 className="text-3xl font-bold">🎟️ TicketTracker</h1>
        <nav className="space-x-6 text-sm">
          <a href="#" className="hover:underline">Home</a>
          <a href="#" className="hover:underline">Browse Events</a>
          <a href="#" className="hover:underline">About</a>
        </nav>
      </header>

      {/* Search Section */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold mb-4">Search Events</h2>
        <div className="flex flex-col sm:flex-row gap-4">
          <input
            type="text"
            placeholder="Enter artist, team, or venue"
            className="w-full px-4 py-2 border rounded"
          />
          <button className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">
            Search
          </button>
        </div>
      </section>

      {/* Trending Events */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold mb-4">Trending Events</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="border p-4 rounded shadow-sm hover:shadow-md transition"
            >
              <h3 className="text-lg font-medium">Event Title #{i}</h3>
              <p className="text-sm text-gray-600">Date • Venue • City</p>
              <p className="text-sm mt-2">From $120</p>
              <button className="mt-3 text-blue-600 hover:underline text-sm">
                View Tickets →
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t pt-6 text-sm text-gray-500 text-center">
        © 2025 TicketTracker. All rights reserved.
      </footer>
    </div>
  );
}
